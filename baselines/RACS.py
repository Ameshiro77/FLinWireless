import numpy as np
from scipy.optimize import minimize
from tianshou.data import Batch
from env.config import *
from baselines.FedAvg import FedAvgPolicy

class RACSAptPolicy(FedAvgPolicy):
    """Reliable and Age-sensitive Client Selection and Adaptive Bandwidth Allocation"""
    def __init__(self, args):
        super().__init__(args)
        self.energy_budgets = np.ones(args.num_clients) *100
        self.aoi_weights = np.ones(args.num_clients)  # Age of Information权重
        self.total_bandwidth = TOTAL_BLOCKS * B
    def calculate_aoi(self, parti_times, current_round):
        """计算数据年龄(Age of Data)"""
        return 1 - (parti_times / current_round)
    
    def learn(self):
        return super().learn()
    def forward(self, obs, state=None, info={}, **kwargs):
        clients = info['clients']
        current_round = info['current_round']
        
        # 1. 计算AoI分数
        aoi_scores = self.calculate_aoi(info['parti_times'], current_round)
        
        # 2. 选择客户端：考虑能量约束、AoI和时间成本
        candidate_scores = []
        for idx in range(self.num_clients):
            if info['parti_times'][idx] >= self.energy_budgets[idx]:
                continue
                
            # 估计时间成本（使用平均带宽）
            avg_rb = self.total_bandwidth / self.num_choose
            clients[idx].set_rb_num(avg_rb)
    
            T = clients[idx].get_cost()[0]  # 获取时间成本
            
            # 综合评分 = AoI * (1/时间成本)
            score = aoi_scores[idx] * (1 / (T + 1e-6))  # 防止除零
            candidate_scores.append((idx, score))
        
        if not candidate_scores:
            return Batch(logits=None, act=None, state=None)
            
        # 按评分排序并选择前K个
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [x[0] for x in candidate_scores[:self.num_choose]]
        
        def smooth_max(times, alpha=10):
            return np.log(np.sum(np.exp(alpha * np.array(times)))) / alpha

        # 3. 自适应带宽分配（与EDCS-Apt相同）
        def objective(n_rb):
            times = []
            for i, idx in enumerate(selected_indices):
                clients[idx].set_rb_num(n_rb[i])
                T = clients[idx].get_cost()[0]
                times.append(T)
            return smooth_max(times, alpha=10)

        
        constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - self.total_bandwidth})
        bounds = [(1, self.total_bandwidth)] * len(selected_indices)
        x0 = [self.total_bandwidth / len(selected_indices)] * len(selected_indices)
        
        res = minimize(objective, x0, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
        print(res)
        allocs = np.floor(res.x).astype(np.int32)
        while sum(allocs) > self.total_bandwidth:
            allocs[np.argmax(allocs)] -= 1
        while sum(allocs) < self.total_bandwidth:
            allocs[np.argmin(allocs)] += 1
        
        bandwidths = allocs / self.total_bandwidth
        action = np.vstack([selected_indices, bandwidths])
        return Batch(logits=action, act=action, state=None)