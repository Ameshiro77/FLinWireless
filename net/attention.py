from torch.distributions import Normal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

import torch.nn.init as init
from torch import Size, Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, norm_type='layer'):
        super().__init__()
        self.norm_type = norm_type
        # self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.norm1 = nn.BatchNorm1d(embed_dim)  # lead hmean to zero.
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        elif norm_type == 'rms':
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(embed_dim)
            self.norm2 = nn.BatchNorm1d(embed_dim)

        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
        # self.norm2 = nn.BatchNorm1d(embed_dim)

    def forward(self, q, k, v):
        attn_out, _ = self.attention.forward(q, k, v)  # MHA 需要 Q, K, V
        # 交换维度，使得 BatchNorm1d 的输入格式正确
        if self.norm_type == 'batch':
            x = (q + attn_out).transpose(1, 2)  # (batch, embed_dim, seq_len)
            x = self.norm1(x).transpose(1, 2)   # 归一化后转回 (batch, seq_len, embed_dim)
            ffn_out = self.ffn(x)
            x = (x + ffn_out).transpose(1, 2)
            x = self.norm2(x).transpose(1, 2)
        else:
            x = self.norm1(q + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
        return x

# ==
# output final probs.


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.query_proj = nn.Linear(hidden_dim, out_dim)
        self.key_proj = nn.Linear(out_dim, out_dim)

    def forward(self, query, key, mask=None):
        # query: (B, 1, H), key: (B, N, H)
        Q = self.query_proj(query)  # (B, 1, H)
        K = self.key_proj(key)      # (B, N, H)

        # 注意力分数: (B, 1, N)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.out_dim)  # Q·K^T / sqrt(d)

        scores = 10 * torch.tanh(scores)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))  # (B, 1, N)

        # probs = F.softmax(scores, dim=-1).squeeze(1)  # (B, N)
        return scores


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # 一般来说，
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim
        assert self.head_dim * nums_head == hidden_dim
        # 一般默认有 bias，需要时刻主意，hidden_dim = head_dim * nums_head，所以最终是可以算成是 n 个矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # X shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)
        batch_size, seq_len, _ = q.size()
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)
        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        # 用 head_dim，而不是 hidden_dim
        # bs , nums_head, seq_len, seq_len
        attention_weight = (q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim))

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 变成 [batch_size, 1, 1, num_clients + 1]
            mask = mask.expand(-1, 4, seq_len, -1)  # 变成 [batch_size, num_heads, num_clients + 1, num_clients + 1]

            # print(attention_weight.shape)
            # print(mask.shape)
            # exit()
            attention_weight = attention_weight.masked_fill(mask == 0, float("-inf"))

        # 第四个维度 softmax
        attention_weight = torch.softmax(attention_weight, dim=3)
        # print(attention_weight.shape)

        output_mid = attention_weight @ v_state

        # reshape to (batch, seq_len, num_head, head_dim)
        output_mid = output_mid.transpose(1, 2).contiguous()
        #  (batch, seq, hidden_dim),
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        # print(output.shape)
        return output
