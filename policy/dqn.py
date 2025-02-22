from tianshou.policy import DQNPolicy, PGPolicy, PPOPolicy,BasePolicy
from tianshou.data import Batch
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class BaseNet(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_shape),  # 输出 n 维连续向量
        ).cuda()

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float).cuda()
        batch = s.shape[0]
        action = self.model(s.view(batch, -1))
        return action, state
        
class BaselinePolicy(BasePolicy):
    def __init__(self, type, model, optim, total_bandwidth, top_n, **kwargs):
        """
        :param type: 策略类型 (DQN, PG, PPO)
        :param model: 基础模型
        :param optim: 优化器
        :param total_bandwidth: 总带宽
        :param top_n: 每次选择的客户端数量
        """
        super().__init__()
        self.type = type
        self.total_bandwidth = total_bandwidth
        self.top_n = top_n  # 每次选择的客户端数量

        # 初始化不同的 tianshou 策略
        if type == "DQN":
            self.policy = DQNPolicy(model, optim, **kwargs)
        elif type == "PG":
            self.policy = PGPolicy(model, optim, **kwargs)
        elif type == "PPO":
            self.policy = PPOPolicy(model, optim, **kwargs)
        else:
            raise ValueError(f"Unsupported policy type: {type}")

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
        # 使用底层策略生成 logits
        logits, state = self.policy.model(batch.obs, state)
        # 转为概率分布
        probs = logits
        print(probs)
        # 带宽分配
        allocations = self.allocate_bandwidth(probs)
        # 返回动作和分布
        return Batch(logits=logits, probs=probs, act=allocations, state=state)

    def learn(self, batch, **kwargs):
        print("\n=======learn!!========\n")
        # 提取 logits 和目标分布
        logits, _ = self.policy.model(batch.obs)
        probs = F.softmax(logits, dim=1)
        act_tensor = torch.tensor(batch.act, dtype=torch.float32).cuda()
        # 自定义损失（例如，使用分配向量计算 MSE 或其他合适的损失）
        loss = F.mse_loss(probs, act_tensor)  # 假设 batch.act 是目标分布
        self.policy.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()
        return {"loss": loss.item()}


    # def update(self, **kwargs):
    #     # 更新底层策略
    #     return self.policy.update(**kwargs)

