"""Formal networks follow manuscript Table 1 and Figures 4/5.

Image values are [0,1]. Prior heads retain Conv-IN-ReLU as published;
the fusion decoder's tanh is mapped to [0,1] before forming a completion.
Historical flow/resampling networks are not used by this implementation.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GatedINReLU(nn.Module):
    def __init__(self, inc, outc, kernel, stride=1, padding=0):
        super().__init__()
        self.feature = nn.Conv2d(inc, outc, kernel, stride, padding)
        self.gate = nn.Conv2d(inc, outc, kernel, stride, padding)
        self.norm = nn.InstanceNorm2d(outc)
    def forward(self, x):
        return F.relu(self.norm(self.feature(x))) * torch.sigmoid(self.gate(x))


class PriorResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Table 1: two 3x3, stride-1, padding-1 residual convolutions.
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                          nn.InstanceNorm2d(channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                          nn.InstanceNorm2d(channels))])
    def forward(self, x, features=None):
        y=x
        for layer in self.layers:
            y=layer(y)
            if features is not None: features.append(y)
        return x+y


class PriorGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, dim=64):
        super().__init__()
        if dim != 64: raise ValueError('Published prior channel width is 64.')
        self.layers = nn.ModuleList([
            GatedINReLU(input_channels,64,7,1,3),
            GatedINReLU(64,128,4,2,1), GatedINReLU(128,128,5,1,2),
            GatedINReLU(128,256,4,2,1), GatedINReLU(256,256,5,1,2),
            GatedINReLU(256,512,4,2,1), PriorResidual(512),
            nn.Upsample(scale_factor=2,mode='nearest'), GatedINReLU(512,256,5,1,2),
            PriorResidual(256), nn.Upsample(scale_factor=2,mode='nearest'),
            GatedINReLU(256,128,5,1,2), PriorResidual(128),
            nn.Upsample(scale_factor=2,mode='nearest'), GatedINReLU(128,64,5,1,2),
            nn.Sequential(nn.Conv2d(64,output_channels,3,1,1),
                          nn.InstanceNorm2d(output_channels),nn.ReLU())])
    def forward(self, x, return_features=False):
        features=[] if return_features else None
        for layer in self.layers:
            if isinstance(layer,PriorResidual): x=layer(x,features)
            else:
                x=layer(x)
                if features is not None and not isinstance(layer,nn.Upsample): features.append(x)
        return (x,features) if return_features else x


class StructureGen(PriorGenerator):
    def __init__(self,input_dim=3,dim=64,**kwargs):
        super().__init__(input_dim*2+1,input_dim,dim)


class TextureGen(PriorGenerator):
    def __init__(self,input_dim=3,dim=64,**kwargs):
        super().__init__(input_dim+2,1,dim)


def gate_norm(feature):
    mean=feature.mean((2,3),keepdim=True)
    std=feature.std((2,3),keepdim=True)+1e-9
    return 5*(2*(feature-mean)/std-1)


class FSTBlock(nn.Module):
    def __init__(self,dim=256,rates=(1,2,4,8)):
        super().__init__()
        def transforms():
            return nn.ModuleList([nn.Sequential(nn.ReflectionPad2d(r),
                nn.Conv2d(dim,dim//4,3,dilation=r),nn.ReLU()) for r in rates])
        self.atb,self.stb,self.ttb=transforms(),transforms(),transforms()
        self.fuse1=nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3))
        self.fuse2=nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3))
        self.gate=nn.Sequential(nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3))
    def forward(self,x1,structure,texture):
        x2=self.fuse1(torch.cat([self.atb[0](x1),self.atb[1](x1),
                                 self.stb[2](structure),self.stb[3](structure)],1))
        # Figure 5: both gates and both residual paths originate from x1.
        g1=1-torch.sigmoid(gate_norm(self.gate(x1)))
        y1=x1*g1+x2*(1-g1)
        x3=self.fuse2(torch.cat([self.ttb[0](texture),self.ttb[1](texture),
                                 self.atb[2](y1),self.atb[3](y1)],1))
        g2=1-torch.sigmoid(gate_norm(self.gate(x1)))
        return x1*g2+x3*(1-g2)


class InpaintingGen(nn.Module):
    def __init__(self,input_dim=3,dim=64,fst_blocks=8,**kwargs):
        super().__init__()
        if dim!=64: raise ValueError('Published fusion encoder width is 64.')
        def encoder(inc):
            return nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(inc,64,7),nn.ReLU(),
                nn.Conv2d(64,128,4,2,1),nn.ReLU(),nn.Conv2d(128,128,5,1,2),nn.ReLU(),
                nn.Conv2d(128,256,4,2,1),nn.ReLU(),nn.Conv2d(256,256,5,1,2),nn.ReLU())
        self.image_encoder=encoder(input_dim*2+1)  # [Id,Sr,M], Figure 4: 7 channels
        self.structure_encoder=encoder(input_dim)
        self.texture_encoder=encoder(1)
        self.middle=nn.ModuleList([FSTBlock() for _ in range(fst_blocks)])
        self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(256,128,5,1,2),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(128,64,5,1,2),nn.ReLU(),nn.Conv2d(64,input_dim,3,1,1),nn.Tanh())
    def forward(self,inputs,rtv_maps,lbp_maps):
        x=self.image_encoder(inputs)
        s=self.structure_encoder(rtv_maps);t=self.texture_encoder(lbp_maps)
        for block in self.middle: x=block(x,s,t)
        return (self.decoder(x)+1)/2,None


class SMPatchDiscriminator(nn.Module):
    """Single-scale spectral-normalized PatchGAN; soft guidance is in the loss.

    Convolutional layout follows the AOT discriminator cited in the manuscript.
    """
    def __init__(self,input_dim=3,dim=64,n_layers=3,**kwargs):
        super().__init__()
        layers=[];inc=input_dim
        for outc,stride in [(dim,2),(dim*2,2),(dim*4,2),(dim*8,1)]:
            layers.extend([spectral_norm(nn.Conv2d(inc,outc,4,stride,1,bias=False)),nn.LeakyReLU(.2)])
            inc=outc
        layers.append(nn.Conv2d(inc,1,4,1,1))
        self.model=nn.Sequential(*layers)
    def forward(self,x):
        return [self.model(x)]  # one-element list preserves the training interface
