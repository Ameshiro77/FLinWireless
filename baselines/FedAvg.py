import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch
import torch
import numpy as np


class FedAvgPolicy(BasePolicy):
    def __init__(self, args):
        super().__init__()
        self.num_clients = args.num_clients
        self.num_choose = args.num_choose

    def learn(self):
        print("Random policy does not require learning")

    def forward(self, obs, state=None, info={}, **kwargs):
        selected_indices = np.random.choice(self.num_clients, self.num_choose, replace=False)
        # weights = np.random.rand(5)
        # proportions = weights / weights.sum()  # float, sum to 1
        bandwidths = np.full(self.num_choose, 1.0/self.num_choose)  # 构造等宽分配数组
        # 构造 2x5 action
        action = np.vstack([selected_indices, bandwidths])
        print(bandwidths)
        return Batch(logits=action, act=action, state=None)
