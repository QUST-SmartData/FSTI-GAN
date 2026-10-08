"""Group processed binary masks by actual missing-pixel ratio.

White/nonzero = missing. Boundaries belong to the next bin, using integer
pixel-count comparisons to avoid floating-point boundary errors. Bins are
[1%,10%), [10%,20%), ..., [50%,60%]. Empty, <1%, and >60% masks are excluded.
Run: python utils/mask_group_split.py --masks MASKS --output groups.csv --size 256
"""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image

BIN_LABELS=('1-10%','10-20%','20-30%','30-40%','40-50%','50-60%')
def group_counts(missing,total):
    if total<=0 or missing<0 or missing>total: raise ValueError('Invalid pixel counts.')
    value=100*int(missing)
    if value<total or value>60*total: return None
    for i,upper in enumerate((10,20,30,40,50,60)):
        if value<upper*total or (i==5 and value==60*total): return BIN_LABELS[i]
    return None
def group_mask(mask):
    arr=np.asarray(mask)
    if arr.ndim!=2: raise ValueError('Pass one processed 2D binary mask.')
    return group_counts(int(np.count_nonzero(arr)),int(arr.size))
def exclusion_reason(missing,total):
    if missing==0:return 'empty mask'
    if 100*missing<total:return 'below 1%'
    if 100*missing>60*total:return 'above 60%'
    return ''
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--masks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--size',type=int,default=256,help='Processed square size; 0 preserves input geometry.')
    a=p.parse_args();root=Path(a.masks)
    paths=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'))
    if not paths:raise ValueError('No mask images.')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['mask','missing_pixels','total_pixels','mask_ratio','group','excluded_reason'])
        for path in paths:
            image=Image.open(path).convert('L')
            if a.size:image=image.resize((a.size,a.size),Image.Resampling.NEAREST)
            mask=np.asarray(image)>0;missing=int(mask.sum());total=int(mask.size)
            writer.writerow([str(path.relative_to(root)),missing,total,missing/total,
                group_counts(missing,total) or '',exclusion_reason(missing,total)])
if __name__=='__main__':main()
