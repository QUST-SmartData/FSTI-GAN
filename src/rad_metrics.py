"""Dataset Rad-FID and paired image-level RadImageNet L2 proxies.

No weights are downloaded and no ImageNet/random fallback is accepted.
The image proxy is not a single-image FID.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from PIL import Image
from scipy import linalg
from scipy.stats import wilcoxon


class RadImageNetFeatures(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError('Supply the actual RadImageNet ResNet-50 checkpoint: ' + str(path))
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('Expected a PyTorch RadImageNet state dictionary.')
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        base = resnet50(weights=None)
        # Official RadImageNet Backbone uses children()[:9], including avgpool.
        self.backbone = nn.Sequential(*list(base.children())[:9])
        if any(k.startswith('conv1.') for k in state):
            names = {'conv1':'0','bn1':'1','layer1':'4','layer2':'5','layer3':'6','layer4':'7'}
            state = {names[k.split('.')[0]]+'.'+k.split('.',1)[1]: v
                     for k,v in state.items() if k.split('.')[0] in names}
        else:
            state = {k.removeprefix('backbone.'):v for k,v in state.items()
                     if not k.startswith(('fc.', 'classifier.'))}
        self.backbone.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """Input N,1,H,W grayscale in [0,1]; output N,2048."""
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError('RadImageNet evaluation expects single-channel medical images.')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('RadImageNet inputs must be finite and lie in [0,1].')
        x = F.interpolate(images.repeat(1,3,1,1), (224,224), mode='bilinear', align_corners=False)
        # Official PyTorch example: (pixel - 127.5)*2/255.
        features = self.backbone(x*2-1).flatten(1)
        if features.shape[1] != 2048:
            raise RuntimeError('Expected 2048-dimensional average-pool features.')
        return features


@torch.inference_mode()
def extract_features(paths, model, batch_size=16, device='cpu'):
    if not paths or batch_size < 1:
        raise ValueError('A nonempty complete image list and positive batch size are required.')
    model = model.to(device).eval()
    features = []
    for start in range(0,len(paths),batch_size):
        # Resize separately so different source dimensions can share a batch.
        batch=[]
        for path in paths[start:start+batch_size]:
            image=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
            tensor=torch.from_numpy(image.copy())[None,None]
            batch.append(F.interpolate(tensor,(224,224),mode='bilinear',align_corners=False)[0])
        features.append(model(torch.stack(batch).to(device)).cpu().numpy().astype(np.float64))
    return np.concatenate(features,axis=0)


def rad_fid(reconstructed, reference):
    """Global Frechet distance using all test images, not a per-image quantity."""
    a,b=np.asarray(reconstructed,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=2048 or b.shape[1]!=2048 or min(len(a),len(b))<2:
        raise ValueError('Rad-FID requires at least two 2048-dimensional vectors per set.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Feature arrays must be finite.')
    ma,mb=a.mean(0),b.mean(0)
    ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    root=linalg.sqrtm(ca@cb)
    if not np.isfinite(root).all():
        eye=np.eye(2048)*1e-6
        root=linalg.sqrtm((ca+eye)@(cb+eye))
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag))>1e-3:
            raise ValueError('Frechet covariance square root has a substantial imaginary component.')
        root=root.real
    return float(max(0,np.sum((ma-mb)**2)+np.trace(ca)+np.trace(cb)-2*np.trace(root)))


def image_l2_proxy(reconstructed, reference):
    a,b=np.asarray(reconstructed,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2048:
        raise ValueError('Paired reconstructed/reference features must both be N x 2048 in identical image order.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Feature arrays must be finite.')
    return np.linalg.norm(a-b,ord=2,axis=1)


def paired_image_wilcoxon(method, baseline):
    """Two-sided paired test on one value per matching test image.

    Call separately for each complete run; do not pool five runs into 5N
    independent images or pretend that global FID has N sample values.
    """
    a,b=np.asarray(method,dtype=np.float64),np.asarray(baseline,dtype=np.float64)
    if a.ndim!=1 or a.shape!=b.shape or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching finite one-dimensional per-image metric arrays.')
    difference=a-b
    if np.all(difference==0):
        return {'statistic':0.,'p_value':1.,'n_images':len(a)}
    result=wilcoxon(difference,alternative='two-sided',zero_method='wilcox',method='auto')
    return {'statistic':float(result.statistic),'p_value':float(result.pvalue),'n_images':len(a)}


def five_run_summary(runs):
    if len(runs)!=5 or len({r['seed'] for r in runs})!=5:
        raise ValueError('Exactly five complete independently seeded runs are required.')
    keys=set(runs[0]['metrics'])
    if any(set(r['metrics'])!=keys for r in runs):
        raise ValueError('Every run must report the same complete-set metric collection.')
    return {key:{'mean':float(np.mean([r['metrics'][key] for r in runs])),
                 'std':float(np.std([r['metrics'][key] for r in runs],ddof=1)),
                 'n_runs':5,'std_unit':'independent_training_run'} for key in sorted(keys)}
