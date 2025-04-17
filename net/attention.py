from alloctor import DirichletPolicy
from torch.distributions import Normal
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
        # self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.norm1 = nn.BatchNorm1d(embed_dim)  # lead hmean to zero.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
        # self.norm2 = nn.BatchNorm1d(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        attn_out, _ = self.attention.forward(q, q, q)  # MHA 需要 Q, K, V
        # 交换维度，使得 BatchNorm1d 的输入格式正确
        # x = (x + attn_out).transpose(1, 2)  # (batch, embed_dim, seq_len)
        # x = self.norm1(x).transpose(1, 2)   # 归一化后转回 (batch, seq_len, embed_dim)
        x = self.norm1(q + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        # x = (x + ffn_out).transpose(1, 2)  # (batch, embed_dim, seq_len)
        # x = self.norm2(x).transpose(1, 2)  # 归一化后转回 (batch, seq_len, embed_dim)
        return x


class SelectEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)  # h = Wx + b
        for layer in self.layers:
            x = layer(x,x,x)
        return x


class AllocEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        #self.embedding = nn.Linear(input_dim, embed_dim)
        #self.layer = EncoderLayer(input_dim, num_heads)
        self.mha = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

    def forward(self, x, select_indices):
        #x = self.embedding(x)  # h = Wx + b
        selected_x = torch.gather(x, dim=1, index=select_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # (B, K, H)
        #print(x.shape,selected_x.shape)
        x = self.mha.forward(selected_x, x, x)
        x = x.view(x.size(0), -1)  #must flatten
        return x

# =======


class SelectDecoder(nn.Module):

    def __init__(self, hidden_dim, num_heads, num_selects):
        super().__init__()
        self.num_selects = num_selects
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)  # b n h
        self.single_head_attn = SingleHeadAttention(hidden_dim)

    def forward(self, encoder_output, is_training):

        batch_size, num_clients, hidden_dim = encoder_output.shape
        mask = torch.zeros(batch_size, num_clients, dtype=torch.bool,
                           device=encoder_output.device)  # true is mask in pth
        selected_clients = torch.zeros(batch_size, self.num_selects, dtype=torch.long, device=encoder_output.device)
        pi = torch.zeros(batch_size, self.num_selects, device=encoder_output.device)
        log_pi = torch.zeros(batch_size, self.num_selects, device=encoder_output.device)

        h_mean = torch.mean(encoder_output, dim=1).unsqueeze(1)

        for step in range(self.num_selects):

            attn_output, _ = self.mha.forward(h_mean, encoder_output, encoder_output,
                                              key_padding_mask=mask)  # d : bs 1 hidden
            # scores = self.single_head_attn(attn_output).squeeze(-1).transpose(0, 1)  # (batch_size, num_clients)
            scores = self.single_head_attn.forward(attn_output, encoder_output, mask).squeeze(1)  # B 1 N -> B N

            # scores[mask] = float('-inf')
            log_probs = F.log_softmax(scores, dim=-1)
            probs = torch.exp(log_probs)
            print(probs)
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

        # print(selected_clients)
        # exit()
        # 返回：
        # selected_clients: 每步选中的动作索引
        # pi: 每步被选中动作的概率
        # joint_entropy: 整个策略序列的联合熵,用于约束
        joint_entropy = -log_pi.sum(dim=-1)
        joint_entropy = first_dist.entropy()
        return selected_clients, pi, joint_entropy


class AllocDecoder(nn.Module):

    def __init__(self, state_dim, num_heads=4, constant=10):
        super().__init__()
        # self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.net = DirichletPolicy(state_dim=state_dim, action_dim=5, hidden_dim=256, constant=constant)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_mu = nn.Linear(hidden_dim, 1)
        # self.fc_std = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, K, H], 每个 client 的状态
        # attn_out, _ = self.attn(x, x, x)   # [B, K, H]
        # h = F.relu(self.fc1(attn_out))     # [B, K, H]
        # mu = self.fc_mu(h).squeeze(-1)     # [B, K]
        # std = F.softplus(self.fc_std(h)).squeeze(-1)  # [B, K]
        # dist = Normal(mu, std)
        # raw_sample = dist.rsample()        # [B, K]
        # allocation = F.softmax(raw_sample, dim=-1)  # 归一化
        # # 估算 log_prob with softmax
        # log_prob = dist.log_prob(raw_sample) - torch.log(allocation + 1e-8)
        # log_prob = log_prob.sum(dim=-1)
        allocation, _, log_prob, entropy = self.net.forward(x, is_training=True)
        return allocation, log_prob, entropy


# class AllocDecoder(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)  # 输入 shape: [B, K, H]
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Mish(),
#             nn.Linear(hidden_dim, 1)  # 每个 client 输出一个 score
#         )

#     def forward(self, selected_h):  # selected_h: [B, K, H]
#         attn_output, _ = self.mha(selected_h, selected_h, selected_h)  # [B, K, H]
#         scores = self.ffn(attn_output).squeeze(-1)  # [B, K, 1] -> [B, K]
#         concentration = F.softplus(scores) + 1e-6  # 确保为正
#         dist = torch.distributions.Dirichlet(concentration)
#         allocation = dist.rsample()  # 采样连续动作
#         logp_alloc = dist.log_prob(allocation)
#         # print("h:", selected_h, "\nattn:", attn_output, "\nscores:", scores)
#         # allocation = F.softmax(scores, dim=-1)  # [B, K]
#         #print("log_alloc", logp_alloc)
#         return allocation, logp_alloc  # 每个 client 分得的比例


class FlexibleDecoder(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
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
