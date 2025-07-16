from torch.distributions import Normal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from attention import *

import torch.nn.init as init
from torch import Size, Tensor


class SelectEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, num_layers, norm_type='layer'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, norm_type='layer') for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)  # h = Wx + b
        for layer in self.layers:
            x = layer(x, x, x)
        return x


class SelectDecoder(nn.Module):

    def __init__(self, hidden_dim, num_heads, num_selects, decoder_type='mask'):
        super().__init__()
        self.num_selects = num_selects
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)  # b n h

        self.decoder_type = decoder_type
        if decoder_type == 'single':
            self.single_head_attn = SingleHeadAttention(hidden_dim * 3, hidden_dim)
            self.init_embed_1 = nn.Parameter(torch.randn(hidden_dim))
            self.init_embed_2 = nn.Parameter(torch.randn(hidden_dim))
        elif decoder_type == 'mask':
            self.single_head_attn = SingleHeadAttention(hidden_dim, hidden_dim)

    def forward(self, encoder_output, is_training):
        # encoder output: b seq h
        batch_size, num_clients, hidden_dim = encoder_output.shape
        mask = torch.zeros(batch_size, num_clients, dtype=torch.bool,
                           device=encoder_output.device)  # true is mask in pytorch
        selected_clients = torch.zeros(batch_size, self.num_selects, dtype=torch.long, device=encoder_output.device)
        pi = torch.zeros(batch_size, self.num_selects, device=encoder_output.device)
        log_pi = torch.zeros(batch_size, self.num_selects, device=encoder_output.device)
        entropy_per_step = torch.zeros(batch_size, self.num_selects, device=encoder_output.device)  # 记录每一步的熵

        h_mean = torch.mean(encoder_output, dim=1).unsqueeze(1)
        h_first = torch.zeros_like(h_mean)
        h_last = torch.zeros_like(h_mean)

        for step in range(self.num_selects):
            if self.decoder_type == 'mask':
                attn_output, _ = self.mha.forward(h_mean, encoder_output, encoder_output,
                                                  key_padding_mask=mask)  # d : bs 1 hidden
                scores = self.single_head_attn.forward(attn_output, encoder_output, mask).squeeze(1)  # B 1 N -> B N
            elif self.decoder_type == 'single':
                if step == 0:
                    h_c = torch.cat([h_mean, self.init_embed_1.unsqueeze(0).expand(batch_size, 1, -1),
                                    self.init_embed_2.unsqueeze(0).expand(batch_size, 1, -1)], dim=-1)
                else:
                    h_c = torch.cat([h_mean, h_last, h_first], dim=-1)  # B,1,2H
                scores = self.single_head_attn.forward(h_c, encoder_output, mask).squeeze(1)

            # scores[mask] = float('-inf')
            log_probs = F.log_softmax(scores, dim=-1)
            probs = torch.exp(log_probs)
            dist = torch.distributions.Categorical(probs)  # 创建分类分布
            entropy_per_step[:, step] = dist.entropy()  # 直接调用entropy()方法计算熵
            # print(probs)
            if step == 0:
                first_dist = torch.distributions.Categorical(probs.clone())

            if is_training:
                sampled_index = torch.multinomial(probs, 1).squeeze(-1)  # (b,1)->(b,) for update mask
            else:
                sampled_index = torch.argmax(probs, dim=-1)

            # 记录概率和对数概率
            pi[:, step] = probs.gather(1, sampled_index.unsqueeze(1)).squeeze(1)
            log_pi[:, step] = log_probs.gather(1, sampled_index.unsqueeze(1)).squeeze(1)

            # 更新掩码
            selected_clients[:, step] = sampled_index
            mask = mask.scatter(1, sampled_index.unsqueeze(1), True)

            # t=1时刻起，要拼接的向量：
            if step == 0:
                h_first = encoder_output[torch.arange(batch_size), sampled_index].unsqueeze(1)  # [B, 1, H]
            h_last = encoder_output[torch.arange(batch_size), sampled_index].unsqueeze(1)  # [B, 1, H]

        # print(selected_clients)
        # exit()
        # 返回：
        # selected_clients: 每步选中的动作索引
        # pi: 每步被选中动作的概率
        # joint_entropy: 整个策略序列的联合熵,用于约束
        # entropy = -log_pi.sum(dim=-1)
        entropy = entropy_per_step.sum(dim=-1)
        # entropy = first_dist.entropy()
        return selected_clients, pi, entropy
