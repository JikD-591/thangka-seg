# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor


import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

# 论文地址：https://arxiv.org/pdf/1910.03151
# 论文：ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)




class DAPPM(BaseModule):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


class PAPPM(DAPPM):
    """PAPPM module in `PIDNet <https://arxiv.org/abs/2206.02066>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.1).
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)
        # self.block = ECAAttention(kernel_size=3)
    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        # for feat in feats:
        #     print(feat.shape)#96
        # feats = [self.block(feat) for feat in feats]
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)

