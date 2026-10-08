import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rad_metrics import five_run_summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('runs',nargs=5);p.add_argument('--save',required=True)
    p.add_argument('--independent-mask-bins',action='store_true',
        help='Aggregate five separately evaluated results for the same mask bin; selected image sets may differ between runs.')
    a=p.parse_args()
    runs=[json.loads(Path(path).read_text(encoding='utf-8')) for path in a.runs]
    names=[list(__import__('numpy').load(Path(path).with_suffix('.features.npz'),allow_pickle=False)['names']) for path in a.runs]
    if a.independent_mask_bins:
        bins={r.get('mask_group') for r in runs}
        if len(bins)!=1 or None in bins:raise ValueError('Select the same explicit processed-mask bin in each run before aggregation.')
        if any(r.get('prediction_kind')!='raw' for r in runs):raise ValueError('Corrected bin metrics require raw predictions in every run.')
        if any(r['n_images']<1 or len(n)!=r['n_images'] for r,n in zip(runs,names)):raise ValueError('Every run must include its complete nonempty selected bin.')
        if any(len(set(n))!=len(n) for n in names):raise ValueError('Duplicate image identities within a run.')
    else:
        if len({r['n_images'] for r in runs})!=1:raise ValueError('Every run must evaluate the same full test set.')
        if any(n!=names[0] for n in names[1:]):raise ValueError('The five runs use different image identities.')
    result={'summary':five_run_summary(runs),'seeds':[r['seed'] for r in runs],
            'paired_tests_by_run':{str(r['seed']):r.get('paired_tests',{}) for r in runs},
            'note':'Bin metrics are calculated within each run first; five run-level values are summarized without pooling images. No aggregated-global-FID significance test is performed.',
            'independent_mask_bins':a.independent_mask_bins,
            'mask_group':runs[0].get('mask_group'),
            'n_images_by_run':{str(r['seed']):r['n_images'] for r in runs},
            'image_identities_by_run':{str(r['seed']):n for r,n in zip(runs,names)}}
    if a.independent_mask_bins:
        actual_ratios=[r['mask_statistics']['mean_actual_mask_ratio'] for r in runs]
        result['measured_mean_mask_ratio_by_run']=dict(zip([str(r['seed']) for r in runs],actual_ratios))
        result['mean_of_run_mean_mask_ratios']=float(__import__('numpy').mean(actual_ratios))
    Path(a.save).parent.mkdir(parents=True,exist_ok=True)
    Path(a.save).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
