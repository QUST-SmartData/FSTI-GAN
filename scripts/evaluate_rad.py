"""Evaluate complete image directories; optionally test a paired baseline.

Run from repo root: python scripts/evaluate_rad.py --reference GT --output OUT
  --weights /path/resnet50_torch.pt --seed 12 --save run12.json --prediction-kind raw
  [--baseline BASELINE --mask-dir USED_MASKS]
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.mask_group_split import group_mask, BIN_LABELS
from src.rad_metrics import RadImageNetFeatures, extract_features, rad_fid, image_l2_proxy, paired_image_wilcoxon

def image_map(directory):
    result={p.name:p for p in Path(directory).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff')}
    if not result:
        raise ValueError('No evaluation images: '+str(directory))
    return result

def read_prediction(path, tensor_directory=None):
    tensor_path = Path(tensor_directory) / (Path(path).name + '.npy') if tensor_directory else None
    if tensor_path:
        if not tensor_path.exists(): raise ValueError('Missing raw prediction tensor: '+str(tensor_path))
        array = np.load(tensor_path, allow_pickle=False).astype(np.float64)
        if array.ndim != 3 or array.shape[0] not in (1,3): raise ValueError('Raw tensor must be CHW with 1 or 3 channels.')
        array = array.transpose(1,2,0)
        if array.shape[2] == 1: array = np.repeat(array,3,axis=2)
        if not np.isfinite(array).all(): raise ValueError('Nonfinite raw prediction.')
        return array
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)/255

def pixel_metrics(paths, references, masks=None, tensor_directory=None):
    out={'L1':[],'PSNR':[],'SSIM':[]}
    if masks:
        out.update(masked_L1=[], outside_L1=[], mask_ratio=[], identity_residual=[])
    for i,(path,ref) in enumerate(zip(paths,references)):
        a=read_prediction(path,tensor_directory)
        b=np.asarray(Image.open(ref).convert('RGB'),dtype=np.float64)/255
        if a.shape!=b.shape or min(a.shape[:2])<11:
            raise ValueError('Paired images must have identical geometry and support 11x11 SSIM: '+str(path))
        error=np.abs(a-b).mean(axis=2)
        out['L1'].append(float(error.mean()*100))
        out['PSNR'].append(float(peak_signal_noise_ratio(b,a,data_range=1)))
        out['SSIM'].append(float(structural_similarity(b,a,data_range=1,win_size=11,channel_axis=-1)))
        if masks:
            mask=np.asarray(Image.open(masks[i]).convert('L'))>0
            if mask.shape!=a.shape[:2] or not mask.any() or mask.all():
                raise ValueError('Mask must have matching geometry and a nonempty missing region.')
            inside=float(error[mask].mean()*100)
            outside=float(error[~mask].mean()*100)
            ratio=float(mask.mean())
            out['masked_L1'].append(inside)
            out['outside_L1'].append(outside)
            out['mask_ratio'].append(ratio)
            out['identity_residual'].append(out['L1'][-1]-ratio*inside-(1-ratio)*outside)
    return out

def main():
    p=argparse.ArgumentParser()
    for name in ('reference','output','weights','save'):p.add_argument('--'+name,required=True)
    p.add_argument('--mask-group',choices=BIN_LABELS,help='Select one actual processed-mask area group; requires --mask-dir.')
    p.add_argument('--prediction-kind', required=True, choices=['raw'], help='Declare raw uncomposited predictions; completed images are not accepted.')
    p.add_argument('--raw-tensor-dir', help='Optional continuous CHW .npy outputs, named IMAGE_NAME.npy; recommended for unquantized pixel metrics.')
    p.add_argument('--baseline-raw-tensor-dir')
    p.add_argument('--baseline');p.add_argument('--mask-dir');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--inception-fid',action='store_true',help='Also compute the existing global Inception FID (its pretrained weights must be available).')
    a=p.parse_args()
    reference,output=image_map(a.reference),image_map(a.output)
    masks=None;groups={};excluded=[]
    if a.mask_group and not a.mask_dir:raise ValueError('--mask-group requires actual processed masks.')
    if a.mask_dir:
        mask_map=image_map(a.mask_dir)
        if not set(mask_map)<=set(reference):raise ValueError('Mask names must have matching references.')
        for name,path in sorted(mask_map.items()):
            mask=np.asarray(Image.open(path).convert('L'))>0
            with Image.open(reference[name]) as truth:
                if mask.shape!=(truth.height,truth.width):
                    raise ValueError('Use masks after final spatial processing: '+name)
            group=group_mask(mask)
            if group is None or (a.mask_group and group!=a.mask_group):excluded.append(name)
            else:groups[name]=group
        if not groups:raise ValueError('No masks in the requested evaluation range.')
        names=sorted(groups)
        if not set(names)<=set(output):raise ValueError('Every eligible reference/mask pair needs a prediction.')
        masks=[mask_map[n] for n in names]
    else:
        if set(reference)!=set(output):raise ValueError('Output and reference filenames must match exactly.')
        names=sorted(reference)
    refs=[reference[n] for n in names];outs=[output[n] for n in names]
    model=RadImageNetFeatures(a.weights)
    fr=extract_features(refs,model,a.batch_size,a.device)
    fo=extract_features(outs,model,a.batch_size,a.device)
    proxy=image_l2_proxy(fo,fr)
    pixel=pixel_metrics(outs,refs,masks,a.raw_tensor_dir)
    result={'seed':a.seed,'n_images':len(names),'metrics':{k:float(np.mean(v)) for k,v in pixel.items() if k not in ('mask_ratio','identity_residual')},
            'proxy_definition':'L2(reconstruction_2048_global_avgpool, matched_truth_2048_global_avgpool)',
            'rad_weights':str(Path(a.weights).resolve()),'ssim_window':11,
            'prediction_kind':'raw', 'pixel_source':'continuous raw tensors' if a.raw_tensor_dir else 'raw image exports (quantized)',
            'group_identity':'mean(r_i*Lmask_i + (1-r_i)*Loutside_i); products of group means are not generally exact',
            'mask_group':a.mask_group,'excluded_masks':excluded,'mask_group_counts':{g:sum(v==g for v in groups.values()) for g in BIN_LABELS}}
    result['metrics']['Rad-FID']=rad_fid(fo,fr)
    if masks:
        ratios=np.asarray(pixel['mask_ratio'])
        inside=np.asarray(pixel['masked_L1']);outside=np.asarray(pixel['outside_L1'])
        result['mask_statistics']={
            'mean_actual_mask_ratio':float(ratios.mean()),
            'mean_weighted_L1_percent':float(np.mean(ratios*inside+(1-ratios)*outside)),
            'max_abs_identity_residual_percent':float(np.max(np.abs(pixel['identity_residual']))),
            'source':'per-image processed masks and paired raw predictions/references',
            'outside_error_is_measured':True,
            'mask_ratio_is_bin_midpoint':False}
    if a.inception_fid:
        if set(names)!=set(reference) or set(names)!=set(output):
            raise ValueError('Global disk Inception FID requires directories containing exactly the selected images. Use prefiltered directories.')
        from scripts.metrics import FID
        result['metrics']['FID']=float(FID().calculate_from_disk(a.output,a.reference))
    if a.baseline:
        base=image_map(a.baseline)
        if not set(names)<=set(base):raise ValueError('Baseline must contain every selected test image.')
        paths=[base[n] for n in names]
        fb=extract_features(paths,model,a.batch_size,a.device)
        bp=image_l2_proxy(fb,fr);bm=pixel_metrics(paths,refs,masks,a.baseline_raw_tensor_dir)
        result['paired_tests']={k:paired_image_wilcoxon(v,bm[k]) for k,v in pixel.items() if k not in ('mask_ratio','identity_residual')}
        result['paired_tests']['Rad_feature_L2_proxy']=paired_image_wilcoxon(proxy,bp)
        result['paired_tests_unit']='image within this independently trained run; no FID p-value is calculated'
    save=Path(a.save);save.parent.mkdir(parents=True,exist_ok=True)
    with save.with_suffix('.images.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['image',*pixel,'Rad_feature_L2_proxy'])
        for i,name in enumerate(names):writer.writerow([name,*[v[i] for v in pixel.values()],proxy[i]])
    save.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(save.with_suffix('.features.npz'),names=np.asarray(names),reference=fr,reconstruction=fo,proxy=proxy)

if __name__=='__main__':main()
