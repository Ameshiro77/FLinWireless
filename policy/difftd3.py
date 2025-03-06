import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy


class DiffusionTD3(BasePolicy):
    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        device: torch.device,
        tau: float = 0.005,
        gamma: float = 0.99,
        training_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # 网络初始化
        self.actor = actor
        self.actor_optim = actor_optim
        self.target_actor = deepcopy(actor)
        self.target_actor.eval()

        self.critic = critic
        self.critic_optim = critic_optim
        self.target_critic = deepcopy(critic)
        self.target_critic.eval()

        # 参数设置
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.training_noise = training_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.rew_norm = reward_normalization
        self.n_step = estimation_step

        # 训练状态
        self.actor_update_count = 0  # 更新计数器

    def compute_target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """带噪声的目标Q值计算"""
        batch = buffer[indices]
        obs_next = to_torch(batch.obs_next, device=self.device, dtype=torch.float32)

        # 目标动作生成（带噪声截断）
        with torch.no_grad():
            target_a = self.target_actor(obs_next)
            noise = torch.randn_like(target_a) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            target_a = torch.clamp(target_a + noise, -1.0, 1.0)

        # 双Q值取最小
        target_q1, target_q2 = self.target_critic(obs_next, target_a)
        return torch.min(target_q1, target_q2).squeeze(dim=-1)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """n-step returns计算"""
        return self.compute_nstep_return(
            batch, buffer, indices,
            self.compute_target_q,
            self.gamma,
            self.n_step,
            self.rew_norm
        )

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        # process_fn -> learn -> post_process_fn
        """从buffer采样更新"""
        if buffer is None:
            return {}
        self.updating = True
        batch, indices = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.updating = False
        return result
    
    def forward(self, batch: Batch, input: str = "obs", **kwargs: Any) -> Batch:
        
        """动作生成（训练时带探索噪声）"""
        obs = batch[input]
        obs_tensor = to_torch(obs, device=self.device, dtype=torch.float32)
        action = self.actor(obs_tensor).cpu().detach().numpy()

        if self.training:  # 训练时添加噪声
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return Batch(act=action)

    def update_critic(self, batch: Batch) -> torch.Tensor:
        """Critic网络更新"""
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)
        act = to_torch(batch.act, device=self.device, dtype=torch.float32)
        target_q = to_torch(batch.returns, device=self.device, dtype=torch.float32)

        # 双Q网络计算
        current_q1, current_q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(current_q1.squeeze(), target_q) + \
            F.mse_loss(current_q2.squeeze(), target_q)

        # 优化步骤
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss

    def update_actor(self, batch: Batch) -> torch.Tensor:
        """Actor网络延迟更新"""
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)

        # 策略梯度计算
        action = self.actor(obs)
        q1, _ = self.critic(obs, action)
        actor_loss = -q1.mean()  # 最大化Q值

        # 优化步骤
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss

    def update_targets(self) -> None:
        """目标网络软更新"""
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """完整更新步骤"""
        # Critic优先更新
        critic_loss = self.update_critic(batch).item()
        result = {'loss/critic': critic_loss}

        # 延迟更新Actor
        self.actor_update_count += 1
        if self.actor_update_count % self.update_actor_freq == 0:
            actor_loss = self.update_actor(batch).item()
            self.update_targets()
            result.update({
                'loss/actor': actor_loss,
                'actor_update_count': self.actor_update_count
            })

        return result


