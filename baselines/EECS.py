import numpy as np
from scipy.optimize import minimize
from tianshou.data import Batch
from collections import Counter
from env.config import *
from baselines.FedAvg import FedAvgPolicy

class EECSPolicy(FedAvgPolicy):
    """
    EECS-Apt: Energy-Efficient Client Selection and Adaptive Bandwidth Allocation
    - Data Quality Score: Q_i = f(skew_i, size_i)
    - Freshness (Age of Data): AoD_i = 1 - (#times selected / current_round)
    - Score = Q_i * AoD_i
    """

    def __init__(self, args):
        super().__init__(args)
        self.total_bandwidth = TOTAL_BLOCKS * B  # e.g., 100
        self.max_rounds_per_client = 100  # simplified energy constraint

    def estimate_label_skew(self, client):
        """计算标签偏差：越接近 1 表示越 skew（不均匀）"""
        labels = [y for _, y in client.local_dataset]
        count = Counter(labels)
        total = sum(count.values())
        probs = np.array([v / total for v in count.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(count))
        skew = 1 - entropy / (max_entropy + 1e-6)
        return skew

    def compute_Qi(self, skew, size, threshold=0.3,
                   beta1=1.0, c1=0.0, gamma1=1.0, beta2=1.0, gamma2=1.0, c2=0.0):
        if skew <= threshold:
            return beta1 * size + c1
        else:
            return (gamma1 * size) / (beta2 + size + gamma2 * size + c2)

    def compute_AoDi(self, parti_times, current_round):
        return 1 - (parti_times / (current_round + 1e-6))

    def forward(self, obs, state=None, info={}, **kwargs):
        clients = info['clients']
        current_round = info['current_round']
        parti_times = info['parti_times']  # shape: [num_clients], count of selected rounds

        Qscores = []
        for i in range(self.num_clients):
            if parti_times[i] >= self.max_rounds_per_client:
                continue
            skew = self.estimate_label_skew(clients[i])
            size = len(clients[i].local_dataset)
            Qi = self.compute_Qi(skew, size)
            AoDi = self.compute_AoDi(parti_times[i], current_round)
            Qscores.append((i, Qi * AoDi))

        if len(Qscores) == 0:
            return Batch(logits=None, act=None, state=None)

        # 1. 选出 Qscore 最大的前 K 个客户端
        Qscores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [i for i, _ in Qscores[:self.num_choose]]

        # 2. 自适应带宽分配（Apt, smooth-max）
        def objective(n_rb):
            times = []
            for i, idx in enumerate(selected_indices):
                clients[idx].set_rb_num(n_rb[i])
                times.append(clients[idx].get_cost()[0])  # 训练 + 通信时间
            # smooth-max 代替 max() 使 SLSQP 可导
            return np.log(np.sum(np.exp(10 * np.array(times)))) / 10

        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - self.total_bandwidth}]
        bounds = [(1, self.total_bandwidth)] * len(selected_indices)
        x0 = [self.total_bandwidth / len(selected_indices)] * len(selected_indices)

        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        allocs = np.floor(res.x).astype(int)

        # 修正总带宽
        while sum(allocs) > self.total_bandwidth:
            allocs[np.argmax(allocs)] -= 1
        while sum(allocs) < self.total_bandwidth:
            allocs[np.argmin(allocs)] += 1

        bandwidths = allocs / self.total_bandwidth
        action = np.vstack([selected_indices, bandwidths])
        return Batch(logits=action, act=action, state=None)
