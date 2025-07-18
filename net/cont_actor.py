from model import DoubleCritic
import torch
from torch import nn
from torch.nn import functional as F
from tianshou.utils.net.common import MLP as SimpleMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from allocator import *
from selector import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class ThreeLayerMLP(torch.nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super().__init__()
        act_fn = nn.Mish if activation == "mish" else nn.ReLU
        self.mid_layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), act_fn(), nn.Linear(hidden_dim, hidden_dim),
                                       act_fn(), nn.Linear(hidden_dim, action_dim), nn.LeakyReLU())

    def forward(self, x):
        logits = self.mid_layer(x)
        return F.softmax(logits, dim=-1)


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch.nn as nn

class ContActor(nn.Module):
    def __init__(self, window_size=5, hidden_size=5, input_dim=4, num_clients=100,
                 embed_dim=128, lstm=False, num_choose=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.num_choose = num_choose

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = SelectEncoder(input_dim=input_dim, embed_dim=embed_dim, num_heads=4, num_layers=3)
        self.LSTMProcessor = ObsProcessor(window_size=window_size, hidden_size=hidden_size, lstm=lstm)
        self.sha = SingleHeadAttention(embed_dim, embed_dim)

        self.alloc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_clients)
        )

    def forward(self, obs_dict, is_training=True, temperature=0.5):
        x = self.LSTMProcessor(obs_dict)
        if len(x.shape) == 2:
            x = x.view(x.shape[0], self.input_dim, self.num_clients).permute(0, 2, 1)  # B x N x input_dim
        x = self.embedding(x)
        x = self.encoder(x)  # B x N x H

        h_mean = x.mean(dim=1).unsqueeze(1)  # B x 1 x H
        scores = self.sha(h_mean, x).squeeze(1)  # B x N，选择分数 logits

        if is_training:
            # 训练时软选择，方便梯度回传
            weights = F.gumbel_softmax(scores, tau=temperature, hard=False)  # B x N，软one-hot
        else:
            # 推理时硬选择，one-hot形式
            weights = F.gumbel_softmax(scores, tau=temperature, hard=True)  # B x N，硬one-hot

        # 选出 top-k 索引
        topk_vals, topk_indices = torch.topk(weights, self.num_choose, dim=1)  # B x K

        # 计算带宽分配
        bandwidth_logits = self.alloc_head(h_mean.squeeze(1))  # B x N
        bandwidth_alloc_all = torch.softmax(bandwidth_logits, dim=-1)  # B x N

        # 取出 top-k 对应的带宽，归一化
        selected_bandwidth = torch.gather(bandwidth_alloc_all, 1, topk_indices)
        selected_bandwidth = selected_bandwidth / (selected_bandwidth.sum(dim=1, keepdim=True) + 1e-12)
        print(topk_indices,selected_bandwidth)
        return weights, topk_indices, selected_bandwidth




if __name__ == "__main__":
    torch.manual_seed(0)
    num_clients = 100
    window_size = 5
    input_dim = 1

    # ====== 构造模拟 obs_dict ======
    # 模拟每个 client 有一个 5×4 的历史窗口状态（B, N, W, D）
    x = data = torch.randn(100, num_clients * input_dim)
    obs_dict = {"client_history": x}  # 如果你的 LSTMProcessor 用这个 key，请对应改名

    # ====== 初始化 actor ======
    actor = ContActor(
        window_size=window_size,
        hidden_size=16,
        input_dim=input_dim,
        num_clients=num_clients,
        embed_dim=64,
        lstm=True,
        num_choose=10
    )

    # ====== 前向 ======
    final_alloc, client_mask, bandwidth_alloc = actor(obs_dict, is_training=True)

    print("final_alloc shape:", final_alloc.shape)        # B x N
    print("client_mask shape:", client_mask.shape)        # B x N
    print("bandwidth_alloc shape:", bandwidth_alloc.shape)  # B x N

    # ====== 假的 critic loss，测试能否反向传播 ======
    dummy_reward = torch.randn_like(final_alloc)  # 假设来自 critic 的梯度信号
    loss = (final_alloc * dummy_reward).sum()     # actor loss: maximize Q
    loss.backward()

    print("Backprop success, actor embedding weight grad norm:",
          actor.embedding.weight.grad.norm().item())

