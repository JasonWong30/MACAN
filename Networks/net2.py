import torch

from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from losses import  SF
# from kornia.losses import ssim_loss
from .TA import Transformer
from .SA import ResidualBlock
from .CA import MultiSpectralAttentionLayer


from losses import Sobelxy
from . import layers as L
from .diversebranchblock import DiverseBranchBlock
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Basic3x3(nn.Module):
    def __init__(self, inplanes, planes, deploy=False):
        super(Basic3x3, self).__init__()
        self.friconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.friconv(x)
        return out


class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, deploy=False):
        super(Basic1x1, self).__init__()
        inter_inplanes = inplanes // 3
        self.output = nn.Sequential(
            nn.Conv2d(inplanes, inter_inplanes , kernel_size=3, stride=1, padding=1),
            nn.Conv2d(inter_inplanes, planes, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.output(x)
        return (out+1)/2
        # return out

class SCM(nn.Module):
    def __init__(self, in_channel, out_channel, dct_w, dct_h, reduction, hg_depth):
        super(SCM, self).__init__()
        self.sa = ResidualBlock(in_channel, out_channel, hg_depth=hg_depth)
        self.ca = MultiSpectralAttentionLayer(out_channel, dct_w, dct_h, reduction, freq_sel_method='top16')

    def forward(self, x):
        y = self.sa(x)
        y = self.ca(y)

        return y

class ResidualSCM(nn.Module):
    def __init__(self, in_channel, out_channel, dct_w, dct_h, reduction, hg_depth):
        super(ResidualSCM, self).__init__()
        self.blocks = nn.ModuleList([
            SCM(in_channel=in_channel, out_channel=out_channel, dct_w=dct_w, dct_h=dct_h,
                reduction=reduction, hg_depth=hg_depth) for i in range(1)]
        )

    def forward(self, x):
        res = x
        for block in self.blocks:
            x = block(x)
        x = x + res
        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channel, out_channel, dct_h, dct_w, reduction=16, depths=2, relu_type='leakyrelu',
                 norm_type='bn', depth=[2, 2, 2, 2], embed_dim=90, heads=[2, 2, 2, 2], img_size=256, split_size=[2, 4],
                 expansion_factor=4., qkv_bias=True, qk_scale=None, drop_rate=0., drop_path_rate=0.1,
                 attn_drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_chk=False, deploy=False, resi_connection='1conv'):
        super(BasicLayer, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}
        self.SCM = ResidualSCM(in_channel, out_channel, dct_w, dct_h, reduction, hg_depth=depths)
        self.DAT = Transformer(depth, out_channel, heads, img_size, split_size, expansion_factor, qkv_bias, qk_scale,
                               drop_rate, drop_path_rate , attn_drop_rate, act_layer, norm_layer, use_chk, resi_connection)
        self.conv = DiverseBranchBlock(out_channel*2, out_channel, kernel_size=3, padding=1, groups=1, deploy=deploy, nonlinear=nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.SCM(x)
        x2 = self.DAT(x)
        x3 = torch.cat((x1, x2), dim=1)

        return self.conv(x3)

class MODEL(nn.Module):
    def __init__(self, in_channel=2, out_channel=32, output_channel=1, dct_h=7, dct_w=7, deploy=False):
        super(MODEL, self).__init__()
        self.convInput = Basic3x3(in_channel, out_channel, deploy)

        self.conv2_1 = DiverseBranchBlock(out_channel, out_channel, kernel_size=3, padding=1, groups=1, deploy=deploy, nonlinear=nn.ReLU(inplace=True))
        self.conv3_1 = DiverseBranchBlock(out_channel, out_channel, kernel_size=3, padding=1, groups=1, deploy=deploy,nonlinear=nn.ReLU(inplace=True))
        self.conv3_2 = DiverseBranchBlock(out_channel, out_channel, kernel_size=3, padding=1, groups=1, deploy=deploy,nonlinear=nn.ReLU(inplace=True))

        self.basicLayer1 = BasicLayer(in_channel=out_channel, out_channel=out_channel, dct_h=dct_h, dct_w=dct_w, deploy=deploy)
        self.basicLayer2 = BasicLayer(in_channel=out_channel, out_channel=out_channel, dct_h=dct_h, dct_w=dct_w, deploy=deploy)
        self.basicLayer3 = BasicLayer(in_channel=out_channel, out_channel=out_channel, dct_h=dct_h, dct_w=dct_w, deploy=deploy)
        self.convolutional_out = Basic1x1(out_channel*3, output_channel)
        # self.ssim = ssim_loss
        # self.sobelconv = Sobelxy()
        # self.L1loss = nn.L1Loss()
        # self.L2loss = nn.MSELoss()

    def forward_loss(self, fused_img, vi, ir):

        l2_loss = self.L2loss(fused_img, vi) + self.L2loss(fused_img, ir)

        y_grad = self.sobelconv(vi)
        ir_grad = self.sobelconv(ir)
        fusimg_grad = self.sobelconv(fused_img)

        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(fusimg_grad, x_grad_joint)

        loss = l2_loss+loss_grad
        loss = loss.mean()
        return loss

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)

        convInput = self.convInput(input)

        bl1 = self.basicLayer1(convInput)

        j = self.conv2_1(convInput)
        bl2 = self.basicLayer2(j)

        k = self.conv3_2(self.conv3_1(convInput))
        bl3 = self.basicLayer3(k)

        bl4 = torch.cat((bl1, bl2, bl3), dim=1)
        out = self.convolutional_out(bl4)

        # loss = self.forward_loss(out, x, y)

        return out


if __name__ == '__main__':
    upscale = 1
    height = 256
    width = 256
    model = MODEL().cuda().eval()
    x = torch.randn((1, 1, height, width)).cuda()
    y = torch.randn((1, 1, height, width)).cuda()
    loss, z = model(x, y)
    print(z.shape)
