import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch

class GreedyPolicy(BasePolicy):
    def __init__(self, data_qualities, args):
        super().__init__()
        self.num_clients = args.num_clients
        self.total_rb = TOTAL_BLOCKS
        self.data_qualities = data_qualities
        
    def forward(self, obs, state=None, info={}, **kwargs):
        batch_size = obs.shape[0]  # 获取 batch_size
        action = torch.zeros((batch_size, self.num_clients), dtype=torch.int)  # 初始化全零的 action 张量

        for i in range(batch_size):
            indices = torch.randperm(self.num_clients)[:5] 
            action[i, indices] = int(self.total_rb / self.num_clients) 
        print(batch_size,action)
        return Batch(logits=action, act=action, state=None)

