import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch


class RandomPolicy(BasePolicy):
    def __init__(self, args):
        super().__init__()
        self.num_clients = args.num_clients
        self.total_rb = TOTAL_BLOCKS  # 假设总带宽资源块为100

    def learn(self):
        print("Random policy does not require learning")

    def forward(self, obs, state=None, info={}, **kwargs):
        batch_size = obs.shape[0]
        actions = torch.zeros((batch_size, self.num_clients), dtype=torch.float32)
        
        for i in range(batch_size):
            # 随机选择1到num_clients个客户端
            num_selected = torch.randint(1, self.num_clients+1, (1,)).item()
            selected = torch.randperm(self.num_clients)[:num_selected]
            
            # 生成随机带宽分配（总和为100）
            bandwidths = torch.rand(num_selected)
            normalized = (bandwidths / bandwidths.sum()) * self.total_rb
            normalized = normalized.round().int()  # 取整
            
            # 处理取整导致的误差
            diff = self.total_rb - normalized.sum()
            if diff != 0:
                normalized[0] += diff  # 将差值加到第一个客户端
            # 确保分配有效
            normalized = torch.clamp(normalized, 0, self.total_rb)
            actions[i, selected] = normalized.float()
        actions = actions.to(torch.int)
        return Batch(logits=actions, act=actions, state=None)
