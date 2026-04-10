import torch
from mmcv.cnn import ConvModule
from torch import nn
from timm.models.layers import DropPath



import warnings

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from mmengine.model import BaseModule
from mmengine.utils import is_tuple_of
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.registry import MODELS



@MODELS.register_module()
class MobileViTBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 transformer_dim=256,
                 ffn_dim=512,
                 n_heads=4,
                 n_transformer_blocks=2,
                 dropout=0.1,
                 drop_path=0.0):
        super().__init__()
        
        # 第一阶段：局部特征提取（CNN）
        self.conv1 = ConvModule(
            in_channels, in_channels, 3, 
            padding=1, groups=in_channels,  # Depthwise卷积
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6'))
        
        # 通道调整
        self.proj = ConvModule(
            in_channels, transformer_dim, 1, 
            norm_cfg=dict(type='BN'),
            act_cfg=None)
        
        # Transformer编码器
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim=transformer_dim,
                num_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                drop_path=drop_path)
            for _ in range(n_transformer_blocks)
        ])
        
        # 第二阶段：特征投影（CNN）
        self.conv2 = ConvModule(
            transformer_dim, out_channels, 1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6'))
        # Skip Connection
        if in_channels != out_channels:
            self.shortcut = ConvModule(
                in_channels, out_channels, 1,
                norm_cfg=dict(type='BN'),
                act_cfg=None)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        # 输入形状: (B, C, H, W)
        residual = x
        x = self.conv1(x)
        x = self.proj(x)
        # ... transformer处理 ...
        x = self.conv2(x)
        return x + self.shortcut(residual)  # 添加残差连接

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
