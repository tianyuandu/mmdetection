import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import BACKBONES
from .resnet import BasicBlock
from ..utils import ResLayer


class kp_module(nn.Module):
    def __init__(self, n, dims, modules,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(kp_module, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = ResLayer(
            BasicBlock,
            curr_dim,
            curr_dim,
            curr_mod,
            norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            curr_dim,
            next_dim,
            curr_mod,
            stride=2,
            norm_cfg=norm_cfg)

        self.low2 = kp_module(n - 1, dims[1:], modules[1:]) if (
            self.n > 1) else ResLayer(
                BasicBlock,
                next_dim,
                next_dim,
                next_mod,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_dim,
            curr_dim,
            curr_mod,
            norm_cfg=norm_cfg,
            reverse=True)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class Hourglass(nn.Module):
    def __init__(self,
                 n,
                 nstack,
                 dims,
                 modules,
                 cnv_dim=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Hourglass, self).__init__()

        self.nstack = nstack

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 128, 256, 1, stride=2, norm_cfg=norm_cfg))

        self.hg_modules = nn.ModuleList([
            kp_module(n, dims, modules) for _ in range(nstack)])

        self.inters = ResLayer(
            BasicBlock, curr_dim, curr_dim, nstack - 1, norm_cfg=norm_cfg)

        self.inters_ = nn.ModuleList([
            ConvModule(curr_dim, curr_dim, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(nstack - 1)])

        self.cnvs = nn.ModuleList([
            ConvModule(curr_dim, cnv_dim, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(nstack)])

        self.cnvs_ = nn.ModuleList([
            ConvModule(cnv_dim, curr_dim, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(nstack - 1)])

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        inter = self.pre(x)
        outs = []

        for ind, layer in enumerate(zip(self.hg_modules, self.cnvs)):
            hg_, cnv_ = layer

            hg = hg_(inter)
            cnv = cnv_(hg)
            outs.append(cnv)

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs


if __name__ == '__main__':
    n = 5
    dims = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    out_dim = 80

    model = Hourglass(
        n=n, nstack=2, dims=dims, modules=modules, out_dim=out_dim).cuda()
    img = torch.rand(4, 3, 511, 511).cuda()
    out = model(img)
