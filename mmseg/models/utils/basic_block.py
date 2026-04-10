# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType





class BasicBlock(BaseModule):
   
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 # n_div: int = 1,
                 # forward_type: str = 'split_cat'
                ):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # self.convP = Partial_conv3(in_channels, n_div, forward_type)
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.downsample = downsample
        # self.convP = Partial_conv3(channels, 2, 'split_cat')
        
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        # out = self.convP(out)
        out = self.conv2(out)
        # print(out.shape)
        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 2

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            3,
            stride,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            channels,
            channels * self.expansion,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        # self.model = MultiScaleSERepViTBlock(inp=channels, oup=channels, kernel_size=3, stride=1)
        # self.model = CoordAtt(channels, channels)
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample
    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.model(out)   
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out

    
    