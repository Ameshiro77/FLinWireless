from tianshou.policy import A2CPolicy
from tianshou.data import Batch
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),  # 加入 LayerNorm
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),  # 加入 LayerNorm
            nn.Linear(128, action_shape),
        ).cuda()

        # 初始化权重
        self.model.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # 使用 Xavier 初始化
            nn.init.constant_(layer.bias, 0.0)     # 将偏置初始化为 0

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32).cuda()
        batch = s.shape[0]
        logits = self.model(s.view(batch, -1))  # 输出 logits
        probs = F.softmax(logits, dim=-1)  # 转为概率分布
        return probs, state


class CriticNet(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 输出单一的状态值 V(s)
        ).cuda()

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32).cuda()
        batch = s.shape[0]
        value = self.model(s.view(batch, -1))  # 输出 V(s)
        return value


class ActorCriticPolicy(A2CPolicy):
    def __init__(self, actor, critic, actor_optim, critic_optim, total_bandwidth, top_n, **kwargs):
        """
        :param actor: Actor 网络
        :param critic: Critic 网络
        :param actor_optim: Actor 的优化器
        :param critic_optim: Critic 的优化器
        :param total_bandwidth: 总带宽
        :param top_n: 每次选择的客户端数量
        """
        super().__init__(actor, critic, actor_optim, critic_optim, **kwargs)
        self.total_bandwidth = total_bandwidth
        self.top_n = top_n

    def allocate_bandwidth(self, probs):
        """实现带宽分配算法：先选 top_n，再线性归一化分配带宽"""
        batch_size, n_clients = probs.shape  # probs 的形状为 (batch_size, n_clients)
        allocations = []
        device = probs.device  # 获取 probs 所在的设备

        for b in range(batch_size):
            prob = probs[b]  # 当前批次的概率分布

            # 1. 根据概率选择前 top_n 个客户端
            sorted_indices = torch.argsort(prob, descending=True)
            selected_indices = sorted_indices[:self.top_n]

            # 2. 对选定客户端的概率进行线性归一化
            selected_probs = prob[selected_indices]
            normalized_probs = selected_probs / selected_probs.sum()

            # 3. 按比例分配带宽
            initial_alloc = torch.floor(normalized_probs * self.total_bandwidth).int()
            remaining = self.total_bandwidth - initial_alloc.sum()

            # 4. 剩余带宽按概率大小分配
            for i in range(remaining):
                initial_alloc[i % self.top_n] += 1

            # 5. 构造当前批次的分配
            alloc = torch.zeros(n_clients, dtype=torch.int, device=device)
            alloc[selected_indices] = initial_alloc
            allocations.append(alloc)

        return torch.stack(allocations)

    def forward(self, batch, state=None, **kwargs):
        # Actor 网络生成概率分布
        probs, state = self.actor(batch.obs, state)
        # 带宽分配
        allocations = self.allocate_bandwidth(probs)
        return Batch(logits=probs, probs=probs, act=allocations, state=state)

    def learn(self, batch, batch_size, repeat, **kwargs):
        # 执行自定义逻辑（如果有）
        print("\n=======A2C learn!!========\n")

        # 调用父类的 learn 方法，确保传入 batch_size 和 repeat
        result = super().learn(batch, batch_size, repeat, **kwargs)

        # 如果需要自定义返回值，可以在这里处理
        return result

