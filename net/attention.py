import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

# ==============
# encoder
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.BatchNorm1d(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        attn_out = self.attention(x, x, x)  # MHA 需要 Q, K, V
        # 交换维度，使得 BatchNorm1d 的输入格式正确
        x = (x + attn_out).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.norm1(x).transpose(1, 2)   # 归一化后转回 (batch, seq_len, embed_dim)
        ffn_out = self.ffn(x)
        x = (x + ffn_out).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.norm2(x).transpose(1, 2)  # 归一化后转回 (batch, seq_len, embed_dim)
        return x


class Encoder(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        # print(x.shape)
        x_global = x.mean(dim=1, keepdim=True)  # 计算全局平均 h̄
        return x
# =======
# decoder


class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Decoder, self).__init__()
        self.end_token = nn.Parameter(torch.randn(hidden_dim))  # 额外的结束标记
        self.mha = MultiHeadAttention(hidden_dim, num_heads)

        self.num_heads = num_heads
        self.dk = hidden_dim // num_heads
        self.single_head_attn = nn.Linear(hidden_dim, 1)  # 单头注意力，用于计算分数

    def forward(self, encoder_output):
        batch_size, num_clients, hidden_dim = encoder_output.shape
        end_token_expanded = self.end_token.unsqueeze(0).expand(batch_size, 1, hidden_dim)
        encoder_output = torch.cat([encoder_output, end_token_expanded], dim=1)  # 添加 end embedding
        avg_embedding = encoder_output.mean(dim=1, keepdim=True)
        mask = torch.zeros(batch_size, num_clients + 1, dtype=torch.bool, device=encoder_output.device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=encoder_output.device)  # 记录哪些 batch 仍需采样
        selected_clients = [[] for _ in range(batch_size)]  # 改为列表，避免 tensor 拼接错误
        while active_mask.any():  
            attn_output = self.mha(encoder_output, encoder_output, encoder_output, attention_mask=~mask)
            scores = self.single_head_attn(attn_output).squeeze(-1)  # (batch_size, num_clients + 1)
            scores[mask] = float('-inf')  # 屏蔽已选项
            # **检查无可选项的 batch**
            invalid_mask = (scores != float('-inf')).sum(dim=-1) == 0  # (batch_size,)
            active_mask[invalid_mask] = False  # 终止无可选项的 batch
            if not active_mask.any():
                break
            # 计算 softmax，但只对有效 batch 进行
            probs = F.softmax(scores, dim=-1)
            sampled_index = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)
            finished = sampled_index == num_clients  # 标记已经选择 end_token 的 batch
            # **这里修改，避免重复添加**
            for i in range(batch_size):
                if active_mask[i]:  # 只存未终止的 batch
                    selected_clients[i].append(sampled_index[i].item())  
            # **关键修正点**：确保 mask 真的屏蔽已选项
            mask.scatter_(1, sampled_index.unsqueeze(-1), True)  
            # **确保 finished batch 不再采样**
            active_mask[finished] = False
            # print(selected_clients)
        return selected_clients


