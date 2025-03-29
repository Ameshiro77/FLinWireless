import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR

#
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR


class DiffusionSAC(BasePolicy):
    """
    Implementation of diffusion-based discrete soft actor-critic policy.
    """

    def __init__(
            self,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            alpha: float = 0.05,
            tau: float = 0.005,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            pg_coef: float = 1.,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._dist_fn = dist_fn
        self._alpha = alpha
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._pg_coef = pg_coef
        self._device = device

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next_ = torch.FloatTensor(batch.obs_next).to(self._device)
        dist = self.forward(
            batch, input="obs_next", model="target_actor").dist
        target_q = dist.probs * self._target_critic.q_min(obs_next_)
        return target_q.sum(dim=-1)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        if buffer is None: return {}
        self.updating = True
        # sample from replay buffer
        batch, indices = buffer.sample(sample_size)
        # calculate n_step returns
        batch = self.process_fn(batch, buffer, indices)
        # update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model_ = self._actor if model == "actor" else self._target_actor
        logits, hidden = model_(obs_), None
        dist = self._dist_fn(logits)
        acts = dist.sample() if self.training else logits.argmax(axis=-1)
        return Batch(logits=logits, act=acts, state=hidden, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act[:, np.newaxis], device=self._device, dtype=torch.long)
        target_q = batch.returns
        current_q1, current_q2 = self._critic(obs_)
        critic_loss = F.mse_loss(current_q1.gather(1, acts_), target_q) \
                      + F.mse_loss(current_q2.gather(1, acts_), target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        return critic_loss

    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = self._to_one_hot(batch.act, self._action_dim)
        acts_ = to_torch(acts_, device=self._device, dtype=torch.float32)
        bc_loss = self._actor.loss(acts_, obs_).mean()
        if update:
            self._actor_optim.zero_grad()
            bc_loss.backward()
            self._actor_optim.step()
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        dist = self.forward(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            q = self._critic.q_min(obs_)
        pg_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update critic network
        critic_loss = self._update_critic(batch)
        # update actor network
        pg_loss = self._update_policy(batch, update=False)
        bc_loss = self._update_bc(batch, update=False) if self._pg_coef < 1. else 0.
        overall_loss = self._pg_coef * pg_loss + (1 - self._pg_coef) * bc_loss
        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()
        # update target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),
            'overall_loss': overall_loss.item()
        }

#

class DiffusionSAC(BasePolicy):

    def __init__(
            self,
            actor: Optional[torch.nn.Module],  # 你的 Diffusion 模型
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            total_bandwidth: int,
            alpha: float = 0.05,
            tau: float = 0.005,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            pg_coef: float = 1.,
            num_choose: int = 5, 
            **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._alpha = alpha
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._pg_coef = pg_coef
        self._device = device
        self._num_choose = num_choose  # 选择 K 个客户端
        self._total_bandwidth = total_bandwidth

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next_ = torch.FloatTensor(batch.obs_next).to(self._device)
        actions_next = self._target_actor.sample(obs_next_)  # 使用扩散模型生成动作
        target_q = self._target_critic(obs_next_, actions_next)
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(batch, buffer, indices, self._target_q, self._gamma, self._n_step,
                                         self._rew_norm)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        self.updating = True
        # sample from replay buffer
        batch, indices = buffer.sample(sample_size)
        # calculate n_step returns
        batch = self.process_fn(batch, buffer, indices)
        # update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result

    def forward(self,
                batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = "obs",
                model: str = "actor") -> Batch:
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model_ = self._actor if model == "actor" else self._target_actor
        actions = model_.sample(obs_)  # 使用扩散模型生成概率分布
        
        # dist = torch.distributions.Distribution(actions)
        allocations = self.allocate_bandwidth(actions)
        return Batch(act=allocations)
        # acts = actions.argmax(axis=-1)
        return Batch(act=acts)
        

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)  # 动作是连续值
        target_q = batch.returns
        current_q = self._critic(obs_, acts_)  # Critic 接受状态和动作 Q(s,a)
        critic_loss = F.mse_loss(current_q, target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        return critic_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        actions = self._actor.sample(obs_)  # 使用扩散模型生成动作
        with torch.no_grad():
            q = self._critic(obs_, actions)  # 计算 Q 值
        entropy = -torch.sum(actions * torch.log(actions + 1e-8), dim=-1)  # 计算熵
        pg_loss = -(q + self._alpha * entropy).mean()  # 策略梯度损失
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, List[float]]:
        # update critic network
        critic_loss = self._update_critic(batch)
        # update actor network
        pg_loss = self._update_policy(batch, update=True)
        # update target networks
        self._update_targets()
        return {'loss/critic': critic_loss.item(), 'loss/policy': pg_loss.item(), 'rew': batch.rew}

    def allocate_bandwidth(self, probs):
        """实现带宽分配算法：先选 top_n，再线性归一化分配带宽"""
        batch_size, n_clients = probs.shape  # probs 的形状为 (batch_size, n_clients)
        allocations = []
        device = probs.device  # 获取 probs 所在的设备

        for b in range(batch_size):
            prob = probs[b]  # 当前批次的概率分布

            # 1. 根据概率选择前 top_n 个客户端
            sorted_indices = torch.argsort(prob, descending=True)
            selected_indices = sorted_indices[:self._num_choose]

            # 2. 对选定客户端的概率进行线性归一化
            selected_probs = prob[selected_indices]
            normalized_probs = selected_probs / selected_probs.sum()

            # 3. 按比例分配带宽
            initial_alloc = torch.floor(normalized_probs * self._total_bandwidth).int()
            remaining = self._total_bandwidth - initial_alloc.sum()

            # 4. 剩余带宽按概率大小分配
            for i in range(remaining):
                initial_alloc[i % self._num_choose] += 1

            # 5. 构造当前批次的分配
            alloc = torch.zeros(n_clients, dtype=torch.int, device=device)
            alloc[selected_indices] = initial_alloc
            allocations.append(alloc)

        return torch.stack(allocations)


if __name__ == "__main__":
    # test
    action_dim = 5  # 假设动作空间维度为10
    state_dim = 10  # 假设状态空间维度为20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义Actor和Critic网络
    # === 指定 actor 以及对应optim ===
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.abspath(__file__)
    # 获取项目根目录（FL）
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(os.path.join(project_root, 'net'))  # noqa
    sys.path.append(project_root)

    # 现在可以正常导入 net.diffusion
    from net.diffusion import Diffusion
    from net.model import MLP

    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=100,
    ).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

    class Critic(torch.nn.Module):

        def __init__(self, state_dim, action_dim):
            super(Critic, self).__init__()
            self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)  # 输入为状态和动作的拼接
            self.fc2 = torch.nn.Linear(256, 1)

        def forward(self, obs, act):
            # 将状态和动作拼接
            x = torch.cat([obs, act], dim=-1)  # 假设状态和动作在最后一维拼接
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    critic = Critic(state_dim, action_dim).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # 初始化 DiffusionSAC 策略
    policy = DiffusionSAC(
        actor=actor,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=device,
        total_bandwidth=100,
        alpha=0.2,
        tau=0.005,
        gamma=0.99,
        reward_normalization=False,
        estimation_step=1,
        lr_decay=True,
        lr_maxt=1000,
        pg_coef=0.5,
        num_choose=5  # 选择 3 个客户端
    )

    # 假设我们有一个ReplayBuffer
    buffer = ReplayBuffer(size=10000)

    # 添加一些随机数据到ReplayBuffer
    # 添加一些随机数据到 ReplayBuffer
    # 添加一些随机数据到 ReplayBuffer
    for _ in range(100):
        obs = np.random.randn(state_dim)
        act = np.random.randn(action_dim)  # 连续动作
        rew = np.random.randn()
        obs_next = np.random.randn(state_dim)
        done = np.random.rand() > 0.95
        buffer.add(
            Batch(
                obs=obs,
                act=act,
                rew=rew,
                obs_next=obs_next,
                done=done,  # 这里添加 'done' 属性，避免 'terminated' 错误
                terminated=done,  # 如果有需要，也可以添加 'terminated' 属性
                truncated=done,  # 如果有需要，也可以添加 'truncated' 属性
            ))

    # 训练循环
    for epoch in range(10):  # 训练 1000 个 epoch
        # 从 ReplayBuffer 中采样
        batch, indices = buffer.sample(64)

        # 更新策略
        result = policy.update(sample_size=64, buffer=buffer)

        # 打印训练日志
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Critic Loss: {result['loss/critic']}, Policy Loss: {result['loss/policy']}")

    # 测试策略
    test_state = torch.randn(1, state_dim).to(device)  # 随机生成一个测试状态
    action = policy.forward(Batch(obs=test_state)).act  # 生成动作
    print("Test State:", test_state)
    print("Generated Action:", action)

##
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from tianshou.data import Collector, VectorReplayBuffer
import torch
from gym import spaces
import torch.nn.functional as F
from tianshou.data import Collector, VectorReplayBuffer
import numpy as np


class DiffusionSAC(BasePolicy):
    """
    Implementation of diffusion-based discrete soft actor-critic policy.
    """

    def __init__(
            self,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            alpha: float = 0.05,
            tau: float = 0.005,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            pg_coef: float = 1.,
            total_bandwidth: int = 100,
            num_choose: int = 5,
            is_not_alloc=False,
            task='fed',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._dist_fn = dist_fn
        self._alpha = alpha
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._pg_coef = pg_coef
        self._device = device
        self._total_bandwidth = total_bandwidth
        self._num_choose = num_choose
        self._is_not_alloc = is_not_alloc
        self._task = task

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next_ = torch.FloatTensor(batch.obs_next).to(self._device)
        act_next_ = self.forward(batch).logits
        target_q = self._target_critic.q_min(obs_next_, act_next_)
        return target_q.sum(dim=-1)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        self.updating = True
        # sample from replay buffer
        batch, indices = buffer.sample(sample_size)
        # exit()
        # calculate n_step returns
        batch = self.process_fn(batch, buffer, indices)
        # update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result

    def forward(self, batch: Batch, input: str = "obs", model: str = "actor") -> Batch:
        if input == None:
            input = "obs"
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model_ = self._actor if model == "actor" else self._target_actor
        logits, hidden = model_(obs_), None  # logits:bs,act_dim
        dist = self._dist_fn(logits)
        if self._task == 'gym':
            return Batch(logits=logits, act=logits, state=hidden, dist=dist)
        elif self._task == 'fed':
            allocs = self.allocate_bandwidth(logits)
            # if self._is_not_alloc:
            #     topk_values, topk_indices = torch.topk(logits, self._num_choose, dim=1)
            #     selection = torch.zeros_like(logits, dtype=torch.int32)
            #     selection.scatter_(1, topk_indices, 1)
            #     return Batch(logits=logits, act=selection, state=hidden, dist=dist)
            # else:
            
            return Batch(logits=logits, act=allocs, state=hidden, dist=dist)
        else:
            raise NotImplementedError

    def _to_one_hot(self, data: np.ndarray, one_hot_dim: int) -> np.ndarray:
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts = to_torch(batch.act, device=self._device, dtype=torch.long)
        target_q = batch.returns

        current_q1, current_q2 = self._critic(obs, acts)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        return critic_loss

    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        bc_loss = self._actor.loss(acts_, obs_).mean()
        if update:
            self._actor_optim.zero_grad()
            bc_loss.backward()
            self._actor_optim.step()
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        act_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        dist = self.forward(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            q = self._critic.q_min(obs_, act_)
        pg_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()

        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update critic network
        critic_loss = self._update_critic(batch)
        # update actor network
        pg_loss = self._update_policy(batch, update=False)
        bc_loss = self._update_bc(batch, update=False) if self._pg_coef < 1. else 0.
        overall_loss = self._pg_coef * pg_loss + (1 - self._pg_coef) * bc_loss
        self._actor_optim.zero_grad()
        overall_loss.backward()
        print(overall_loss)
        for name, param in self._actor.named_parameters():
            print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean()}!")
        self._actor_optim.step()
        # update target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),
            'overall_loss': overall_loss.item(),
            'rew': batch.rew
        }

    def allocate_bandwidth(self, probs):
        """实现带宽分配算法：先选 top_n，再线性归一化分配带宽"""
        batch_size, n_clients = probs.shape  # probs 的形状为 (batch_size, n_clients)
        allocations = []
        selects = []
        device = probs.device  # 获取 probs 所在的设备

        for b in range(batch_size):
            prob = probs[b]  # 当前批次的概率分布

            # 1. 根据概率选择前 top_n 个客户端
            sorted_indices = torch.argsort(prob, descending=True)
            selected_indices = sorted_indices[:self._num_choose]

            # 2. 对选定客户端的概率进行线性归一化
            selected_probs = prob[selected_indices]
            normalized_probs = selected_probs / selected_probs.sum()

            # 3. 按比例分配带宽
            initial_alloc = torch.floor(normalized_probs * self._total_bandwidth).int()
            remaining = self._total_bandwidth - initial_alloc.sum()

            # 4. 剩余带宽按概率大小分配
            for i in range(remaining):
                initial_alloc[i % self._num_choose] += 1

            # 5. 构造当前批次的分配
            alloc = torch.zeros(n_clients, dtype=torch.int, device=device)
            alloc[selected_indices] = initial_alloc
            allocations.append(alloc)
            selects.append(selected_indices)
        return torch.stack(allocations)


if __name__ == "__main__":
    import sys  # noqa
    import os  # noqa

    current_dir = os.path.abspath(__file__)  # noqa
    project_root = os.path.dirname(os.path.dirname(current_dir))  # noqa
    sys.path.append(os.path.join(project_root, 'net'))  # noqa
    sys.path.append(os.path.join(project_root, 'env'))  # noqa
    sys.path.append(project_root)

    # 现在可以正常导入 net.diffusion
    from env.dataset import FedDataset
    from env.client import get_args, init_attr_dicts, init_clients
    from env.models import MNISTResNet
    from env.FedEnv import FederatedEnv, make_env
    from net.diffusion import Diffusion
    from net.model import *

    # === 初始化环境 ===
    args = get_args()
    dataset = FedDataset(args)
    model = MNISTResNet()
    attr_dicts = init_attr_dicts(args.num_clients)
    clients = init_clients(args.num_clients, model, dataset, attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
        print(f"Client {i} has {len(subset)} samples")
    data_distribution = dataset.get_data_distribution()
    print("Data distribution:")
    for client_id, dist in data_distribution.items():
        print(f"Client {client_id}: {dist},samples num:{sum(dist)}")
    print("clients attr:")
    for i in range(args.num_clients):
        print(f"client {i} initialized,attr:{attr_dicts[i]}")

    env, train_envs, test_envs = make_env(args, dataset, clients, model=model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 指定 actor 以及对应optim ===
    # 计算 state_shape
    state_dim = 0
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            state_dim += 1  # Discrete 空间是标量
        elif isinstance(space, spaces.Box):
            state_dim += np.prod(space.shape)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    action_dim = 10
    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=100,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)

    # === 指定 critic 以及对应optim ===
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化 DiffusionSAC 策略 ===
    policy = DiffusionSAC(
        actor=actor,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        dist_fn=torch.distributions.Categorical,
        device=device,
        alpha=0.2,
        tau=0.005,
        gamma=0.99,
        reward_normalization=False,
        estimation_step=1,
        lr_decay=True,
        lr_maxt=1000,
        pg_coef=0.5,
    )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    for i in range(10):  # 训练总数
        train_collector.collect(n_step=2)
        # 使用采样出的数据组进行策略训练
        losses = policy.update(2, train_collector.buffer)
