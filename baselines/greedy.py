import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch
import numpy as np


class GreedyPolicy(BasePolicy):
    def __init__(self, args):
        super().__init__()
        self.num_clients = args.num_clients
        self.total_rb = TOTAL_BLOCKS  # 假设总带宽资源块为100

        self.num_choose = args.num_choose

    def forward(self, obs, state=None, info={}, **kwargs):
        data_sizes = torch.tensor(info['data_sizes'])
        data_qualities = torch.tensor(info['data_qualities'])  # 客户端数据质量
        gain = data_sizes / data_qualities
        _, topk_indices = torch.topk(gain, k=self.num_choose)
        selected_indices = topk_indices.numpy()  # 客户端索引
        bandwidths = np.full(self.num_choose, 1.0/self.num_choose)  # 构造等宽分配数组
        action = np.vstack([selected_indices, bandwidths])
        # 返回Batch对象（如果是单样本，直接返回数组；如果是batch，返回列表）
        return Batch(logits=action, act=action, state=None)

    def learn(self):
        print("Random policy does not require learning")
