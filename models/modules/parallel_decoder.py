

import torch
from torch import Tensor
from torch.nn import (
    Module, init, Sequential, Linear, ReLU, BatchNorm1d, LayerNorm, Dropout, functional as F, MultiheadAttention
)
from typing import List, Optional, Tuple
from collections import OrderedDict

#这个模块是多头自注意力机制的实现。它接收查询（query）、键（key）、
# 值（value）作为输入，并计算注意力分数，输出加权后的上下文表示。
class MultiHeadAttention(Module):
    def __init__(
            self, dim: int, num_heads: int,
            dropout: float = 0.0, bias: bool = True, dim_key: Optional[int] = None, dim_value: Optional[int] = None,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.qry_proj = Linear(dim, dim, bias=bias)
        self.key_proj = Linear(dim_key or dim, dim, bias=bias)
        self.val_proj = Linear(dim_value or dim, dim, bias=bias)
        self.out_proj = Linear(dim, dim, bias=bias)
        self.dropout = Dropout(dropout)

    def reset_parameters(self) -> None:
        for proj in [self.qry_proj, self.key_proj, self.val_proj, self.out_proj]:
            init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                init.zeros_(proj.bias)

    def forward(
            self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        N, L, E = query.shape
        N, S, _ = key.shape
        q = self.qry_proj(query).view(N, L, self.num_heads, self.dim_head).transpose(1, 2)  # (N, M, L, H)
        k = self.key_proj(key).view(N, S, self.num_heads, self.dim_head).transpose(1, 2)  # (N, M, S, H)
        v = self.val_proj(value).view(N, S, self.num_heads, self.dim_head).transpose(1, 2)  # (N, M, S, H)
        scores = torch.matmul(q * self.scale, k.transpose(2, 3))  # (N, M, L, S)
        if mask is not None:
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill_(mask, float('-inf')).view(N, 1, 1, S)
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(N, L, E)  # (N, L, E)
        context = self.out_proj(context)
        return context, weights

#这个模块是一个常见的前馈神经网络，包含两个全连接层（fc1 和 fc2）以及一个 ReLU 激活函数。
# 它用来对输入进行非线性变换。 dropout：防止过拟合。
class FeedForwardNetwork(Sequential):
    def __init__(self, dim: int, dim_hidden: int = 2048, dropout: float = 0.0):
        super().__init__(OrderedDict([
            ('fc1', Linear(dim, dim_hidden)),
            ('relu', ReLU(inplace=True)),
            ('fc2', Linear(dim_hidden, dim)),
            ('dropout', Dropout(dropout)),
        ]))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc2.bias)

#emb_align：这个函数用于将查询嵌入序列与记忆对齐。
# 它通过在每个序列末尾填充零来确保所有查询序列长度相同，便于后续的并行处理。
def emb_align(
        queries: Tensor,
        num_queries_per_memory: List[int],
        value: float = 0.,
        num_memories_first: bool = True
) -> Tensor:
    """
    Arguments:
         queries: the shape is (sum(num_queries_per_memory), vector_dim)
         num_queries_per_memory:
         value: the padding value
         num_memories_first:
            if true: (sum(num_queries_per_memory), vector_dim) -> (max_num, num_memories, vector_dim)
            if false: (sum(num_queries_per_memory), vector_dim) -> (num_memories, max_num, vector_dim)
    """
    max_num = max(num_queries_per_memory)
    return torch.stack([
        F.pad(query_sequence, (0, 0, 0, max_num - num_queries), 'constant', value)
        for query_sequence, num_queries in zip(queries.split(num_queries_per_memory), num_queries_per_memory)
    ], dim=1 - num_memories_first)

#emb_dealign：这个函数将经过对齐后的查询嵌入恢复成原始的形状，去除之前填充的部分。
def emb_dealign(queries: Tensor, num_embs_per_sequence: List[int], num_memories_first: bool = True) -> Tensor:
    queries = queries if num_memories_first else queries.transpose(0, 1)
    return torch.cat([
        query_sequence[:num_queries]
        for query_sequence, num_queries in zip(queries, num_embs_per_sequence)
    ])


class ParallelDecoder(Module):
    def __init__(
            self, dim: int, num_heads: int,
            dim_ffn: int = 2048, dropout: float = 0., dim_memory: Optional[int] = None #定义场景的维度是 2048
    ):
        super().__init__()
        self.norm1 = BatchNorm1d(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = BatchNorm1d(dim)
        self.ffn1 = FeedForwardNetwork(dim, dim_ffn, dropout=dropout)
        self.ffn2 = FeedForwardNetwork(dim, dim_ffn, dropout=dropout)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, dropout=dropout, dim_key=dim_memory or dim, dim_value=dim_memory or dim
        )

    def reset_parameters(self) -> None:
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.ffn1.reset_parameters()
        self.ffn2.reset_parameters()
        self.cross_attn.reset_parameters()

    def forward(self, queries: Tensor, memories: Tensor, num_queries_per_memory: List[int]) -> Tensor:
        queries = queries + self.ffn1(self.norm1(queries))
        queries = emb_align(queries, num_queries_per_memory, num_memories_first=True)
        queries = self.cross_attn(query=self.norm2(queries), key=memories, value=memories)[0]  # k和v是 场景的特征  q是行人的特征
        queries = emb_dealign(queries, num_queries_per_memory, num_memories_first=True)
        queries = queries + self.ffn2(self.norm3(queries))
        return queries


if __name__ == '__main__':
    model = ParallelDecoder(1024, 8, 2048, 0.0, 2048)  #输入维度1024 8个头  ffn的维度2048  dropout是0  memory是2048维度
    print(model(torch.randn(10, 1024), torch.randn(4, 49, 2048), [1, 2, 3, 4]).shape)

'''
功能总结：
查询与记忆交互：该模型的主要功能是对查询和记忆进行交互，利用多头注意力机制将查询序列与记忆序列融合，生成最终的查询表示。
对齐和去对齐：为了实现高效的并行处理，模型通过 emb_align 和 emb_dealign 对查询嵌入进行对齐与去对齐。
多头注意力和前馈网络：通过多个多头注意力层和前馈神经网络，增强模型的表达能力，使其能够捕捉到查询和记忆之间的复杂关系。
适用场景：
这个模型的架构类似于 Transformer，但针对特定的查询和记忆对齐的需求进行了定制，适用于需要处理序列数据（如图像描述生成、
语音识别、时间序列预测等）的问题，尤其是当输入查询和记忆的数量或长度不一致时。
'''