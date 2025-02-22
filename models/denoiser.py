

import torch
from torch import Tensor
from torch.nn import (
    Module, init, Sequential, Linear, BatchNorm1d, Identity, Flatten
)
from torchvision.ops.misc import Permute
from typing import List, Optional
from collections import OrderedDict


class CrossAttention(Module):
    def __init__(self, pool: Module, embedder: Optional[Module], norm: Optional[Module], decoder: Module):
        super().__init__()
        self.pool = pool
        self.flatten = Flatten(start_dim=2, end_dim=-1)
        self.permute = Permute([0, 2, 1])
        self.embedder = embedder or Identity()
        self.norm = norm or Identity()
        self.decoder = decoder

    def reset_parameters(self) -> None:
        getattr(self.embedder, 'reset_parameters', lambda: None)()
        getattr(self.norm, 'reset_parameters', lambda: None)()
        self.decoder.reset_parameters()

    def forward(self, embeddings: Tensor, noise_maps: Tensor, num_embs_per_map: List[int]) -> Tensor:
        noise_maps = self.pool(noise_maps)  ###噪声图就是 经过三个卷积后的 特征 然后在进行 池化 permute 和2d位置编码 LN
        noise_maps = self.flatten(noise_maps)
        noise_maps = self.permute(noise_maps)
        noise_maps = self.embedder(noise_maps)  #embedder = LearnablePositionEmbedder2D
        noise_maps = self.norm(noise_maps)  #这里的noise map就是输入到 cross-attention里的特征了 也就是 K和V的向量
        return embeddings + self.decoder(embeddings.detach(), noise_maps, num_embs_per_map)  # decoder = ParallelDecoder
    ##decoder的输入就是 embeddings.detach() 防止梯度回传  还需要结合 noise map 最后得到是queries 然后才是和embedding想加操作

# 功能总结：
# CrossAttention 的目标是将噪声图信息与输入的嵌入结合，生成调整后的嵌入表示。
# 它首先处理噪声图，通过池化、扁平化、嵌入和归一化等步骤，将噪声图转换为可以与嵌入进行交互的形式，
# 最后通过交叉注意力机制生成输出。


class LinearProjection(Module):
    def __init__(self, dim: int, memery_channels: int):
        super().__init__()
        self.projector = Sequential(OrderedDict([
            ('fc', Linear(memery_channels, dim, bias=True)),
            ('bn', BatchNorm1d(dim)),
        ]))

    def reset_parameters(self) -> None:
        init.normal_(self.projector.fc.weight, std=0.01)
        init.zeros_(self.projector.fc.bias)
        self.projector.bn.reset_parameters()

    def forward(self, embeddings: Tensor, noise_maps: Tensor, num_embs_per_map: List[int]) -> Tensor:
        noise_vectors = torch.amax(noise_maps, dim=(2, 3))
        noise_vectors = torch.repeat_interleave(
            noise_vectors,
            torch.tensor(num_embs_per_map, device=noise_vectors.device),
            dim=0,
        )
        return embeddings + self.projector(noise_vectors)
#这个类通过线性变换对噪声图进行处理，然后将处理后的结果加到嵌入上。