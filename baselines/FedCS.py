import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch
import numpy as np
import torch


class FedCSPolicy(BasePolicy):
    def __init__(self, args,min_time):
        super().__init__()
        self.total_rb = TOTAL_BLOCKS  # 总资源块为100
        self.deadline = min_time
        self.max_rb_per_client = TOTAL_BLOCKS  # 单客户端最多分配多少资源块（如100）

    def get_min_rb(self, client):
        """枚举资源块（整数），直到满足时间不超过 deadline"""
        for rb in range(1, self.max_rb_per_client + 1):
            client.set_rb_num(rb)
            # print(client.get_cost()[0])
            if client.get_cost()[0] <= self.deadline:
                return rb
        return None  # 无法满足deadline

    def forward(self, obs, state=None, info={}, **kwargs):
        clients = info['clients']  # client对象列表
        min_rb_list = []

        # 1. 计算每个客户端的最小资源块需求
        for i, client in enumerate(clients):
            min_rb = self.get_min_rb(client)
            print(min_rb)
            # exit()
            if min_rb is not None:
                min_rb_list.append((i, min_rb))
        # 2. 按所需资源块升序排序（贪婪：先挑更省资源的）
        min_rb_list.sort(key=lambda x: x[1])

        selected_indices = []
        bandwidths = []
        used_rb = 0

        for client_id, rb in min_rb_list:
            if used_rb + rb <= self.total_rb:
                selected_indices.append(client_id)
                bandwidths.append(rb/self.total_rb)
                used_rb += rb
            else:
                break

        # 3. 构造动作：2 x K
        action = np.vstack([selected_indices, bandwidths])
        return Batch(logits=action, act=action, state=None)

    def learn(self, batch, **kwargs):
        return super().learn(batch, **kwargs)
