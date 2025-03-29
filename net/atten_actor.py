import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

# 每次时间步的输入处理


import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.total_bandwidth = 100  # 总带宽资源

        self.encoder = Encoder(input_dim, hidden_dim, num_heads, num_layers)
        self.selection_decoder = Decoder(hidden_dim, num_heads)  # 客户端选择解码器


        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 带宽必须为正
        )

        # 上下文聚合
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        encoder_output = self.encoder(x)  # [B, num_clients, hidden_dim]
        selected_indices = self.selection_decoder(encoder_output)  # [B, T] (变长)
        print(selected_indices)
        # 3. 提取被选客户端的隐藏状态
        selected_hidden = self._extract_selected_hidden(encoder_output, selected_indices)  # List[[T_i, hidden_dim]]

        # 4. 动态带宽分配
        bandwidths = self._allocate_bandwidth(selected_hidden, selected_indices)  # [B, num_clients]

        return {
            'selected_clients': selected_indices,
            'bandwidth_allocation': bandwidths
        }

    def _extract_selected_hidden(self, encoder_output, indices):
        """提取被选客户端的隐藏状态（自动跳过结束标记）"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        extracted = []

        for i in range(batch_size):
            # 去掉结束标记
            valid_indices = [idx for idx in indices[i] if idx < self.num_clients]

            if valid_indices:
                indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
                extracted.append(encoder_output[i, indices_tensor])
            else:
                extracted.append(torch.empty(0, encoder_output.size(-1), device=device))

        return extracted

    def _allocate_bandwidth(self, selected_hidden, selected_indices):
        """基于注意力机制的带宽分配"""
        batch_size = len(selected_hidden)
        bandwidths = torch.zeros(batch_size, self.num_clients, device=selected_hidden[0].device)

        for i in range(batch_size):
            if len(selected_hidden[i]) > 0:
                # 计算全局上下文
                global_context = self.context_proj(selected_hidden[i].mean(dim=0))  # [hidden_dim]

                # 计算每个被选客户端的注意力权重
                query = global_context.unsqueeze(0).expand(len(selected_hidden[i]), -1)
                keys = selected_hidden[i]
                attention_scores = torch.sum(query * keys, dim=1)  # [T_i]

                # 归一化到总和为100
                raw_allocation = F.softmax(attention_scores, dim=0) * self.total_bandwidth
                allocated = raw_allocation.round().int()

                # 处理取整误差
                diff = self.total_bandwidth - allocated.sum()
                allocated[0] += diff

                # 写入结果
                bandwidths[i, selected_indices[i][:len(allocated)]] = allocated

        return bandwidths

    def _select_clients(self, logits):
        """从logits采样客户端序列（带结束标记）"""
        # 实现采样逻辑（如top-k或多项式采样）
        pass


if __name__ == "__main__":
    num_clients = 10
    input_dim = 4
    hidden_dim = 128
    num_heads = 4
    num_layers = 3
    budget = 3

    model = PolicyNetwork(input_dim, hidden_dim, num_heads, num_layers, num_clients)
    client_features = torch.randn(8, num_clients, input_dim)
    result = model(client_features)
    print(result)
