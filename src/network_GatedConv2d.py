import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.spectral_norm import spectral_norm
# from spatial_correlation_sampler import spatial_correlation_sample
import functools
from .resample2d import Resample2d
import torchvision.utils as vutils
import numpy as np
from torch.nn import init

def _init_conv_layer(conv, activation, mode='fan_out'):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       a=activation.negative_slope,
                                       nonlinearity='leaky_relu',
                                       mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       nonlinearity='relu',
                                       mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)


class Discriminator(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_layers=3,
                 norm='none', activ='lrelu', pad_type='reflect', use_sn=True):
        super(Discriminator, self).__init__()

        self.model = nn.ModuleList()
        self.model.append(Conv2dBlock(input_dim, dim, 4, 2, 1, 'none', activ, pad_type, use_sn=use_sn))
        dim_in = dim
        for i in range(n_layers - 1):
            dim_out = min(dim * 8, dim_in * 2)
            self.model.append(DownsampleResBlock(dim_in, dim_out, 'none', activ, pad_type, use_sn))
            dim_in = dim_out

        self.model.append(Conv2dBlock(dim_in, 1, 1, 1, activation='none', use_bias=False, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MultiDiscriminator(nn.Module):
    def __init__(self, **parameter_dic):
        super(MultiDiscriminator, self).__init__()
        self.model_1 = Discriminator(**parameter_dic)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_2 = Discriminator(**parameter_dic)

    def forward(self, x):
        pre1 = self.model_1(x)
        pre2 = self.model_2(self.down(x))
        return [pre1, pre2]


class StructureGen(nn.Module):
    def __init__(self, input_dim=3, dim=32, n_res=1, activ='relu', norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()

        # 使用GatedConv2dWithActivation替换Conv2dBlock
        self.input_layer = GConv(input_dim * 2 + 1, dim, 5, 1, 2)
        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(dim, 2 * dim))]

        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(2 * dim, 4 * dim))]

        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(4 * dim, 8 * dim))]
        dim = 8 * dim

        # 使用GatedDeConv2dWithActivation替换Conv2dBlock
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim, dim // 2)))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim // 2, dim // 4)))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim // 4, dim // 8)))]

        self.content_param += [GConv(dim // 2, dim // 2, 3, 1)]
        self.content_param += [GConv(dim // 4, dim // 4, 3, 1)]
        self.content_param += [GConv(dim // 8, dim // 8, 3, 1)]

        self.image_net = Get_image(dim // 8, input_dim)

    def forward(self, inputs):
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out


class InpaintingGen(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=2, activ='relu',
                 norm_flow='ln', norm_conv='in', pad_type='reflect', use_sn=True):
        super(InpaintingGen, self).__init__()

        self.lbp_column = LbpColumn(input_dim, dim, n_res, activ,
                                    norm_flow, pad_type, use_sn)
        self.conv_column = ConvColumn(input_dim, dim, n_res, activ,
                                      norm_conv, pad_type, use_sn)

    def forward(self, inputs, rtv_maps, lbp_maps):
        # print('inputs_size', inputs.size())
        lbp_map = self.lbp_column(inputs)
        images_out = self.conv_column(inputs, lbp_map, rtv_maps, lbp_maps)
        return images_out, lbp_map



# class TextureGen(nn.Module):
#     def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu',
#                  norm='in', pad_type='reflect', use_sn=True):
#         super(TextureGen, self).__init__()
#
#         self.down_sample = nn.ModuleList()
#         self.up_sample = nn.ModuleList()
#         self.content_param = nn.ModuleList()
#         # input_dim, output_dim, kernel_size, stride,padding = 0, norm = 'none', activation = 'relu', pad_type = 'zero', dilation = 1,use_bias = True, use_sn = False):
#         # self.input_layer = Conv2dBlock(input_dim * 2 + 1, dim, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn)
#         self.input_layer = Conv2dBlock(5, dim, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn)
#         self.down_sample += [nn.Sequential(
#             Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
#             Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]
#
#         self.down_sample += [nn.Sequential(
#             Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
#             Conv2dBlock(4 * dim, 4 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]
#
#         self.down_sample += [nn.Sequential(
#             Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))]
#         dim = 8 * dim
#         # content decoder
#         # num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False
#         self.up_sample += [(nn.Sequential(
#             ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
#             nn.Upsample(scale_factor=2),
#             Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]
#
#         self.up_sample += [(nn.Sequential(
#             ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
#             nn.Upsample(scale_factor=2),
#             Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]
#
#         self.up_sample += [(nn.Sequential(
#             ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
#             nn.Upsample(scale_factor=2),
#             Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]
#
#         self.content_param += [Conv2dBlock(dim // 2, dim // 2, 5, 1, 2, norm, activ, pad_type)]
#         self.content_param += [Conv2dBlock(dim // 4, dim // 4, 5, 1, 2, norm, activ, pad_type)]
#         self.content_param += [Conv2dBlock(dim // 8, dim // 8, 5, 1, 2, norm, activ, pad_type)]
#
#         # self.image_net = Get_image(dim // 8, input_dim)
#         self.image_net = Get_image(dim // 8, 1)
#
#     def forward(self, inputs):
#         x0 = self.input_layer(inputs)
#         x1 = self.down_sample[0](x0)
#         x2 = self.down_sample[1](x1)
#         x3 = self.down_sample[2](x2)
#
#         u1 = self.up_sample[0](x3) + self.content_param[0](x2)
#         u2 = self.up_sample[1](u1) + self.content_param[1](x1)
#         u3 = self.up_sample[2](u2) + self.content_param[2](x0)
#
#         images_out = self.image_net(u3)
#         return images_out


class TextureGen(nn.Module):
    def __init__(self, input_dim=3, dim=32, n_res=1, activ='relu', norm='in', pad_type='reflect', use_sn=True):
        super(TextureGen, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()

        # 使用GatedConv2dWithActivation替换Conv2dBlock
        self.input_layer = GConv(input_dim * 2 + 1, dim, 5, 1, 2)
        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(dim, 2 * dim))]

        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(2 * dim, 4 * dim))]

        self.down_sample += [nn.Sequential(
            GDownsamplingBlock(4 * dim, 8 * dim))]
        dim = 8 * dim

        # 使用GatedDeConv2dWithActivation替换Conv2dBlock
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim, dim // 2)))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim // 2, dim // 4)))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            # nn.Upsample(scale_factor=2),
            GUpsamplingBlock(dim // 4, dim // 8)))]

        self.content_param += [GConv(dim // 2, dim // 2, 3, 1)]
        self.content_param += [GConv(dim // 4, dim // 4, 3, 1)]
        self.content_param += [GConv(dim // 8, dim // 8, 3, 1)]

        self.image_net = Get_image(dim // 8, 1)

    def forward(self, inputs):
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out



class ConvColumn(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
                 norm='ln', pad_type='reflect', use_sn=True):
        super(ConvColumn, self).__init__()

        self.down_sample = nn.ModuleList()
        self.down_sample_rtv = nn.ModuleList()
        self.down_sample_lbp = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        self.down_sample += [nn.Sequential(
            Conv2dBlock(input_dim * 2 + 1, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]

        self.down_sample_rtv += [nn.Sequential(
            Conv2dBlock(3, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]

        self.down_sample_lbp += [nn.Sequential(
            Conv2dBlock(1, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(2 * dim, 4 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn))]
            # Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))]

        self.middle += nn.Sequential(*[FSTBlock(256, [1, 2, 4, 8]) for _ in range(8)])

        dim = 8 * dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]

        self.up_sample += [(nn.Sequential(
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),

            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            Get_image(dim // 8, input_dim)))]

        self.resample16 = Resample2d(16, 1, sigma=4)
        self.resample4 = Resample2d(4, 1, sigma=2)

    def forward(self, inputs, lbp_map, rtv_maps, lbp_maps):
        """x1:128 x2:256"""

        x1 = self.down_sample[0](inputs)
        x2 = self.down_sample[1](x1)
        r1 = self.down_sample_rtv[0](rtv_maps)
        r2 = self.down_sample[1](r1)
        l1 = self.down_sample_lbp[0](lbp_maps)
        l2 = self.down_sample[1](l1)
        x3 = self.middle[0](x2, r2, l2)
        """x3:256 lbp_maps:2"""
        lbp_fea = self.resample_image(x1, lbp_map)
        # print('x2.size()=', x2.size())
        # print('x3.size()=', x3.size())
        # print('lbp_fea.size()=', lbp_fea.size())
        # u1 = torch.cat((self.up_sample[0](x3), lbp_fea), 1)
        u1 = torch.cat((x3, lbp_fea), 1)
        # print('u1_fea.size()=', u1.size())
        # u2 = self.up_sample[0](u1)
        # print('u2_fea.size()=', u2.size())
        images_out = self.up_sample[1](u1)
        return images_out
    # image_out.size() = (8,3,256,256)

    def resample_image(self, img, lbp):
        output16 = self.resample16(img, lbp)
        output4 = self.resample4(img, lbp)
        outputs = torch.cat((output16, output4), 1)
        return outputs


class LbpColumn(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
                 norm='in', pad_type='reflect', use_sn=True):
        super(LbpColumn, self).__init__()

        self.down_sample_flow = nn.ModuleList()
        self.up_sample_flow = nn.ModuleList()

        self.down_sample_flow.append(nn.Sequential(
            Conv2dBlock(input_dim * 2 + 1, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
        self.down_sample_flow.append(nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
        self.down_sample_flow.append(nn.Sequential(
            Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(4 * dim, 4 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
        self.down_sample_flow.append(nn.Sequential(
            Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(8 * dim, 8 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
        dim = 8 * dim

        # content decoder
        self.up_sample_flow.append(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim, dim // 2, 6, 2, 2, norm=norm, activation=activ)))

        self.up_sample_flow.append(nn.Sequential(
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 2, dim // 4, 6, 2, 2, norm=norm, activation=activ)))

        self.location = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

    def forward(self, inputs):
        f_x1 = self.down_sample_flow[0](inputs)
        f_x2 = self.down_sample_flow[1](f_x1)
        f_x3 = self.down_sample_flow[2](f_x2)
        f_x4 = self.down_sample_flow[3](f_x3)

        f_u1 = torch.cat((self.up_sample_flow[0](f_x4), f_x3), 1)
        f_u2 = torch.cat((self.up_sample_flow[1](f_u1), f_x2), 1)
        flow_map = self.location(f_u2)
        return flow_map


##################################################################################
# Basic Blocks
##################################################################################

class GConv(nn.Module):
    """Implements the gated 2D convolution introduced in
       `Free-Form Image Inpainting with Gated Convolution`(Yu et al., 2019)
    """

    def __init__(self, cnum_in, cnum_out,
                 ksize, stride=1, padding='auto', rate=1,
                 activation=nn.ELU(),
                 bias=True, gated=True):
        super().__init__()

        padding = rate*(ksize-1)//2 if padding == 'auto' else padding
        self.activation = activation
        self.cnum_out = cnum_out
        # num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2*cnum_out
        num_conv_out = 2*cnum_out if gated else cnum_out
        self.conv = nn.Conv2d(cnum_in,
                              num_conv_out,
                              kernel_size=ksize,
                              stride=stride,
                              padding=padding,
                              dilation=rate,
                              bias=bias)

        self.norm = nn.InstanceNorm2d(cnum_out)

        _init_conv_layer(self.conv, activation=self.activation)

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.gated = gated

    def forward(self, x):
        """
        Args:

        """
        if not self.gated: return self.conv(x)

        x = self.conv(x)
        x, y = torch.split(x, self.cnum_out, dim=1)
        x = self.norm(x)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y

        return x

# ----------------------------------------------------------------------------

class GDeConv(nn.Module):
    """Upsampling (x2) followed by convolution"""

    def __init__(self, cnum_in, cnum_out, padding=1):
        super().__init__()

        self.conv = GConv(cnum_in, cnum_out, ksize=3, stride=1,
                          padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest',
                          recompute_scale_factor=False)
        x = self.conv(x)
        return x

# ----------------------------------------------------------------------------

class GDownsamplingBlock(nn.Module):
    """Strided convolution (s=2) followed by convolution (s=1)"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()

        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, ksize=3, stride=2)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x

# ----------------------------------------------------------------------------

class GUpsamplingBlock(nn.Module):
    """Upsampling (x2) followed by two convolutions"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x


class Get_image(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh'):
        super(Get_image, self).__init__()
        self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1,
                                padding=1, pad_type='reflect', activation=activation)

    def forward(self, x):
        return self.conv(x)


class FSTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(FSTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x, rtv_maps, lbp_maps):
        # out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = []
        for i in range(len(self.rates)):
            if i == 2 or i == 3:
                tmp_out = self.__getattr__(f'block{str(i).zfill(2)}')(rtv_maps)
            else:
                tmp_out = self.__getattr__(f'block{str(i).zfill(2)}')(x)
            out.append(tmp_out)
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        x1 = x * (1 - mask) + out * mask

        out = []
        for i in range(len(self.rates)):
            if i == 0 or i == 1:
                tmp_out = self.__getattr__(f'block{str(i).zfill(2)}')(lbp_maps)
            else:
                tmp_out = self.__getattr__(f'block{str(i).zfill(2)}')(x1)
            out.append(tmp_out)
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class DilationBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(DilationBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 2, norm=norm, activation=activation, pad_type=pad_type, dilation=2)]
        model += [Conv2dBlock(dim, dim, 3, 1, 4, norm=norm, activation=activation, pad_type=pad_type, dilation=4)]
        model += [Conv2dBlock(dim, dim, 3, 1, 8, norm=norm, activation=activation, pad_type=pad_type, dilation=8)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1,
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if use_sn:
            self.conv = spectral_norm(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu'):
        super(TransConv2dBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'in_affine':
            self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.transConv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

    def forward(self, x):
        x = self.transConv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class DownsampleResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(DownsampleResBlock, self).__init__()
        self.conv_1 = nn.ModuleList()
        self.conv_2 = nn.ModuleList()

        self.conv_1.append(Conv2dBlock(input_dim, input_dim, 3, 1, 1, 'none', activation, pad_type, use_sn=use_sn))
        self.conv_1.append(Conv2dBlock(input_dim, output_dim, 3, 1, 1, 'none', activation, pad_type, use_sn=use_sn))
        self.conv_1.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_1 = nn.Sequential(*self.conv_1)

        self.conv_2.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_2.append(Conv2dBlock(input_dim, output_dim, 1, 1, 0, 'none', activation, pad_type, use_sn=use_sn))
        self.conv_2 = nn.Sequential(*self.conv_2)

    def forward(self, x):
        out = self.conv_1(x) + self.conv_2(x)
        return out

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = functools.partial(SwitchNorm2d)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

