"""Soft-mask least-squares terms separated into discriminator/generator updates.

Real=1, completed fake target=blur(1-M), as in manuscript Eqs. 3/9/14.
Gaussian 71x71, sigma=10 follows the cited AOT implementation convention.
"""
import torch
from torch import nn
import torch.nn.functional as F

class SoftMaskAdversarialLoss(nn.Module):
    def __init__(self,kernel_size=71,sigma=10):
        super().__init__()
        axis=torch.arange(kernel_size,dtype=torch.float32)-(kernel_size-1)/2
        gaussian=torch.exp(-axis.square()/(2*sigma*sigma))
        gaussian=gaussian/gaussian.sum()
        self.register_buffer('kernel',torch.outer(gaussian,gaussian)[None,None])
        self.padding=kernel_size//2
    def forward(self,prediction,masks,is_real,for_dis):
        prediction=F.interpolate(prediction,size=masks.shape[-2:],mode='bilinear',align_corners=True)
        if for_dis:
            if is_real: target=torch.ones_like(prediction)
            else:
                known=1-masks
                target=F.conv2d(F.pad(known,(self.padding,)*4,mode='reflect'),self.kernel)
            return (prediction-target).square().mean()
        return ((prediction-1).square()*masks).mean()
