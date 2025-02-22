#需要修改默认文件里面 _C.MODEL.REID.DIM_IDENTITY = 1024
#### 动态选择频域模块 外加如何去融合的embedding块
import pdb

import torch
from torch import Tensor
from torch.nn import init, Module, ModuleList, Sequential, Linear, BatchNorm1d, AdaptiveMaxPool2d, Flatten
from typing import List, Dict, Callable, Optional, Any
from collections import OrderedDict

from models.modules.cbam import RobustCBAM
from models.modules.drop_path import DropPath
from models.modules.MDAF import MDAF
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


import matplotlib.pyplot as plt
import numpy as np
import os

#平均池化
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


# --------------------------------------------------------------------------------

##基础卷积 conv+bn+GELU
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

#自适应平均池化
class Gap(nn.Module):
    def __init__(self, in_channel, mode) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif mode[0] == 'test':
            self.gap = AvgPool2d(base_size=75)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d * self.fscale_d[None, :, None, None]
        return x_d + x_h

##设计的类似res的结构
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter
       ##定义的两个 动态核 一个3 一个5
        self.dyna = dynamic_filter(in_channel // 2, mode) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel // 2, mode, kernel_size=5) if filter else nn.Identity()

        ##定义的3个 动态核 一个3 一个5  一个7
        # self.dyna = dynamic_filter(in_channel // 3, mode) if filter else nn.Identity()
        # self.dyna_2 = dynamic_filter(in_channel // 3, mode, kernel_size=5) if filter else nn.Identity()
        # self.dyna_3 = dynamic_filter(in_channel // 3, mode, kernel_size=7) if filter else nn.Identity()

        self.localap = Patch_ap(mode, in_channel // 2, patch_size=2)
        self.global_ap = Gap(in_channel // 2, mode)

    def reset_parameters(self):
        # 初始化conv1和conv2的权重  这样也是为了稳定训练过程
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)

        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)   #通道分割 将张量 out 在通道维度上分为两部分
            #k3, k5, k7 = torch.chunk(out, 3, dim=1)  # 通道分割 将张量 out 在通道维度上分为三部分
            out_k3 = self.dyna(k3)  #前一半通道
            out_k5 = self.dyna_2(k5)  #后一半通道   每部分的形状为 [B, C/2, H, W]。
            #out_k7 = self.dyna_3(k7)  假如设计为3分支时候
            # 打印形状
            print(f"low_filter.shape (k3): {low_filter_shape_k3}")
            print(f"low_part.shape (k3): {low_part_shape_k3}")
            print(f"low_filter.shape (k5): {low_filter_shape_k5}")
            print(f"low_part.shape (k5): {low_part_shape_k5}")
            out = torch.cat((out_k3, out_k5), dim=1)
            #假设3分支时候
            #out = torch.cat((out_k3, out_k5, out_k7), dim=1)  #将动态滤波后的结果 out_k3 和 out_k5 在通道维度上拼接，恢复 out 的形状为 [B, C, H, W]

        non_local, local = torch.chunk(out, 2, dim=1)  #再次将 out 在通道维度上分为两部分
        non_local = self.global_ap(non_local) #全局特征  应用全局池化操作，通常用于提取非局部的全局特征
        local = self.localap(local)  #局部特征   对 local 应用局部池化操作，保留更多的局部信息
        out = torch.cat((non_local, local), dim=1)  #将经过非局部和局部池化的特征 non_local 和 local 在通道维度拼接
        out = self.conv2(out) #对融合后的特征应用进一步卷积，提取最终的特征
        return out + x

##动态滤波器的功能  对高 低频 进行权重分配和融合    在这里进行分组g的测试 可以。  2 4 8 16 32
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, mode, kernel_size=3, stride=1, group=32):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  #两个可学习的参数 分别对应高频和低频 用于动态调整滤波
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False) #1*1卷积 用于生成 低频滤波器   g*k*k大小 输出
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)  #BN稳定训练
        self.act = nn.Softmax(dim=-2)  #归一化 低频滤波器的权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)  #对输入进行边界填充，保持输入张量的空间大小不变

        self.ap = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化  压缩空间维度 和 提取全局信息
        self.modulate = SFconv(inchannels, mode)  ###进行调制

    def forward(self, x):
        identity_input = x  #方便进行残差 链接
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)  #全局平均池化 生成空间紧凑的低频特征 通过1*1卷积得到 动态滤波器 并归一化

        n, c, h, w = x.shape   #将特征 转换成 滑动窗口模式 unfold函数  提取k*k的局部感受野 特征   n  g  c/g  K*k  hw
        #每一组特征（self.group）分离出 c // self.group 通道，并展开为 kernel_size ** 2 滑动窗口特征，覆盖整个特征图的空间位置（h * w）。
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)  #转化后的特征 表示每个组的 局部特征。
        #c1: 滤波器通道数。p, q: 滤波器的空间大小（通常是池化后的特征尺寸）。
        n, c1, p, q = low_filter.shape  #调整 低通滤波器的形状  与展开后的 输入特征（滑动窗口的特征） 相匹配  并通过softmax归一化  确保滤波权重的 有效性。  n c/k*k k*k p*q
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)  #在通道维度拆分为每个窗口特征的通道数 在第 2 维增加一个维度，用于匹配滑动窗口特征 x
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)  #展开后的特征x  与低频滤波器 按照通道维度进行加权求和   在reshape回 nchw形状。

        out_high = identity_input - low_part  #高频特征 通过 残差计算得到  原始特征减去低频特征
        out = self.modulate(low_part, out_high)  #调用 SFconv 调制低频和高频的权重 并融合生成最后的输出特征。
        return out

##选择调制模块 Select Frequency Conv
class SFconv(nn.Module):
    def __init__(self, features, mode, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)  #这个FC操作是Conv 这是一个 1x1 卷积操作（虽然被称为 “FC”），它的作用类似于全连接层，用于减少通道数，从 features 压缩到 d。
        #这里使用了M个1*1卷积层，每个卷积层对应一个频率组（如高频和低频）作用是从压缩后的特征维度d恢复到原始维度features。
        self.fcs = nn.ModuleList([])  #也是Conv
        for i in range(M):  #M：通道划分的子组数（默认 2，表示两个子组：高频和低频）。
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1) #用于归一化通道权重
        self.out = nn.Conv2d(features, features, 1, 1, 0)  #对融合后的高频和低频特征进行调整，生成最终的输出

        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif mode[0] == 'test':
            self.gap = AvgPool2d(base_size=75)

    def forward(self, low, high):
        emerge = low + high  #sum在一起
        emerge = self.gap(emerge) #GAP操作

        fea_z = self.fc(emerge)  #分成两部分了  1*1 卷积 压缩通道 通过 self.fc 将特征通道数压缩到 d

        high_att = self.fcs[0](fea_z)    #用 self.fcs 中的卷积层分别为高频和低频生成通道权重
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)  #拼接 将高频和低频的权重拼接起来

        attention_vectors = self.softmax(attention_vectors)  #softmax 通过 Softmax 操作进行归一化，确保高频和低频的权重总和为 1
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)    #分成两部分 将归一化后的权重拆分回高频权重和低频权重

        fea_high = high * high_att #高 低频 分别和权重注意力  高频特征乘以其对应的权重
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)  #融合 将加权后的高频和低频特征相加，形成融合特征   self.out 进一步调整输出特征
        return out


class Patch_ap(nn.Module):  #对特征图进行 非局部特征提取 和 局部特征增强，然后融合两者的输出以提升最终特征表达能力
    def __init__(self, mode, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1, 1)) #自适应平均池化
        elif mode[0] == 'test':
            self.gap = AvgPool2d(base_size=75)  #全局池化

        self.patch_size = patch_size  # in_channel // 2, patch_size=2
        self.channel = inchannel * patch_size ** 2
        self.h = nn.Parameter(torch.zeros(self.channel))  #h和l定义 分别调整 高频和低频的特征
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size) #第一行：将输入特征图按 patch_size 分块，形成 p1 × w1 和 p2 × w2 的局部特征。
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size) #第二行：将每个块展平成单通道特征，方便后续操作

        low = self.ap(patch_x) #使用自适应池化提取每个分块的低频特征
        high = (patch_x - low) * self.h[None, :, None, None]  #通过减去低频特征 low 提取高频分量，并对高频分量施加权重  h
        out = high + low * self.l[None, :, None, None]  #将高频分量和经过权重  l  调整的低频分量相加，得到增强的局部特征
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)  #将分块特征恢复成原始输入特征图的形状

        return out


class GlobalFeatureEmbedding(Sequential):
    def __init__(self, in_channels: int, dim_out: int):
        super().__init__(OrderedDict([
            ('gmp', AdaptiveMaxPool2d(1)),
            ('flatten', Flatten(start_dim=1, end_dim=-1)),
            ('linear', Linear(in_channels, dim_out)),
            ('norm', BatchNorm1d(dim_out)),
        ]))

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)



class Embedder(nn.Module):
    def __init__(
            self,
            in_feat_names: List[str],
            in_channels_list: List[int],
            dim_out: int = 1024,
            emb_type: Callable[..., nn.Module] = ResBlock,
            extra_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_feat_names = in_feat_names
        self.dim_out = dim_out

        # ResNet block for res4
        self.res4_encoder = emb_type(in_channels_list[0], in_channels_list[0], **extra_cfg or {})  # 不改变ResNet逻辑

        # 卷积层用于将res4的通道数升维到与res5一致
        self.res4_adjust = nn.Conv2d(in_channels_list[0], in_channels_list[1], kernel_size=1, stride=1, bias=False)

        # MDAF block for fusion
        self.MDAF_L = MDAF(in_channels_list[1], num_heads=8, LayerNorm_type='WithBias')

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def reset_parameters(self) -> None:
        self.res4_encoder.apply(self._reset_weights)
        self.res4_adjust.apply(self._reset_weights)

    @staticmethod
    def _reset_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)

    def forward(self, feat_maps: Dict[str, Tensor]) -> Tensor:
        # Process res4 with ResBlock
        res4_feat = feat_maps[self.in_feat_names[0]]
        #print(f"Input shape for res4: {res4_feat.shape}")
        processed_res4 = self.res4_encoder(res4_feat)  # 保留原ResBlock的逻辑
        #print(f"Output shape after ResBlock for res4: {processed_res4.shape}")
        # 调整res4的通道数
        adjusted_res4 = self.res4_adjust(processed_res4)
        #print(f"Output shape after adjustment for res4: {adjusted_res4.shape}")

        # Get res5 features
        res5_feat = feat_maps[self.in_feat_names[1]]
        #print(f"Input shape for res5: {res5_feat.shape}")

        # 调整res4的空间维度以匹配res5
        adjusted_res4 = F.interpolate(adjusted_res4, size=res5_feat.shape[-2:], mode='bilinear', align_corners=False)
        #print(f"Output shape after interpolation for res4: {adjusted_res4.shape}")

        # Fuse adjusted res4 and res5 using MDAF
        fused_features = self.MDAF_L(adjusted_res4, res5_feat)
        #print(f"Output shape after MDAF: {fused_features.shape}")

        # Global Average Pooling
        embeddings = self.global_avg_pool(fused_features).view(fused_features.size(0), -1)
        #print(f"Output shape after Global Average Pooling: {embeddings.shape}")
        return embeddings




# Input shape for res4: torch.Size([72, 512, 24, 12])
# Output shape after ResBlock for res4: torch.Size([72, 512, 24, 12])
# Output shape after adjustment for res4: torch.Size([72, 1024, 24, 12])
# Input shape for res5: torch.Size([72, 1024, 12, 6])
# Output shape after interpolation for res4: torch.Size([72, 1024, 12, 6])
# Output shape after MDAF: torch.Size([72, 1024, 12, 6])
# Output shape after Global Average Pooling: torch.Size([72, 1024])


# 测试代码
if __name__ == '__main__':

    _type = ResBlock,

    #512+1024维度的 convext 1536维度 效果最好目前
    model = Embedder(
        ['res4', 'res5'], [1024, 2048], 1024, _type,
        mode=['train']
    ).to('cuda')


    model.reset_parameters()
    sample_feat_maps = {
        'res4': torch.randn(24, 1024, 24, 12, device='cuda'),
        'res5': torch.randn(24, 2048, 12, 6, device='cuda')
    }
    print(model(sample_feat_maps).shape)




























