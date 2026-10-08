"""Launch fresh independent training runs, never resume across seeds."""
import argparse, subprocess, sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--name',required=True);p.add_argument('--path',default='./results')
    p.add_argument('--sr-checkpoint');p.add_argument('--tr-checkpoint')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    # Train SR/TR independently first. For fusion, use a {seed} template to pair
    # each seed with its own independently trained prior networks.
    for seed in (12,42,88,123,2026):
        name=f'{a.name}_seed{seed}'
        destination=Path(a.path)/name
        if destination.exists():raise FileExistsError('Refusing to reuse an existing run: '+str(destination))
        args=[sys.executable,str(root/'train.py'),'--config',a.config,'--name',name,'--path',a.path,'--seed',str(seed)]
        for flag,value in (('--sr-checkpoint',a.sr_checkpoint),('--tr-checkpoint',a.tr_checkpoint)):
            if value:args.extend([flag,value.format(seed=seed)])
        subprocess.run(args,cwd=root,check=True)
