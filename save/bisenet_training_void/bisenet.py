# ===========================================================================
# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
# URL: https://arxiv.org/abs/1808.00897 by Changqian Yu, Jingbo Wang, et al.
# ===========================================================================

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from resnet import Resnet18
from torch.nn import BatchNorm2d

# Spatial Path (to preserve spatial information and generate high-resolution features)
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, k_size = 3, stride = 1, padding = 1, *args, **kwargs):
        """
        Module Conv + BatchNorm + ReLU in BiSeNet Spatial Path.

        Args:
            in_chan (int): Number of input channels.
            out_chan (int): Number of output channels.
            k_size (int): Dimension of the kernel. Default 3.
            stride (int): Stride. Default 1.
            padding (int): Padding. Default 1.
        """
        super(ConvBNReLU, self).__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels = in_chan, 
            out_channels = out_chan, 
            kernel_size = k_size, 
            stride = stride, 
            padding = padding,
            bias = False
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace = True)
        self.init_weight()

    def forward(self, input):
      input = self.conv(input)
      input = self.bn(input)
      input = self.relu(input)
      return input
        
    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a = 1)
                if not layer.bias is None:
                    nn.init.constant_(layer.bias, 0)

class UpSample(nn.Module): 

    def __init__(self, num_chan, factor = 2, *args, **kwargs):
        super(UpSample, self).__init__(*args, **kwargs)
        out_chan = num_chan * factor * factor
        self.proj = nn.Conv2d(num_chan, out_chan, kernel_size = 1, stride = 1, padding = 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, input):
        feature = self.proj(input)
        feature = self.up(feature)
        return feature
    
    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain = 1.0)

class SpatialPath(nn.Module): 
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, k_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, k_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, k_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, k_size=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        feature = self.conv_out(feature)
        return feature
    
    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if not layer.bias is None:
                    nn.init.constant_(layer.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
                elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                    nowd_params += list(module.parameters())
        return wd_params, nowd_params


# Context Path (to obtain sufficient receptive field)
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, k_size=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class ContextPath(nn.Module): 
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, k_size=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, k_size=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, k_size=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.0)
        self.up16 = nn.Upsample(scale_factor=2.0)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FeatureFusionModule(nn.Module): 
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, k_size=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        #  self.conv1 = nn.Conv2d(out_chan,
        #          out_chan//4,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.conv2 = nn.Conv2d(out_chan//4,
        #          out_chan,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        #  atten = self.conv1(atten)
        #  atten = self.relu(atten)
        #  atten = self.conv2(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, k_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class BiSeNet(nn.Module):
    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return (feat_out,)
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params