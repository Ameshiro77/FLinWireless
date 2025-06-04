from copy import deepcopy
from typing import Any, Callable, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from tianshou.data import Collector, VectorReplayBuffer
import torch
from gym import spaces
import torch.nn.functional as F
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
import numpy as np
from tianshou.data import Batch, to_torch_as


class TD3Policy(BasePolicy):
    """
    refer to tianshou api
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        double_critic: torch.nn.Module,  # double critic
        critic_optim: torch.optim.Optimizer,
        device: torch.device,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: float = 0.02,  # GASSIAN SIGMA
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        alpha: float = 2.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        # about tasks:
        total_bandwidth: int = 100,
        num_choose: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)\

        # actor
        self.actor = actor
        self.actor_optim = actor_optim
        self.target_actor = deepcopy(actor)
        self.target_actor.eval()

        # double critic
        self.critic = double_critic
        self.critic_optim = critic_optim
        self.target_critic = deepcopy(double_critic)
        self.target_critic.eval()

        self.device = device
        self.tau = tau  # soft update
        self.gamma = gamma  # discount factor
        self.explore_noise = exploration_noise
        self.policy_noise = policy_noise

        self.update_actor_freq = update_actor_freq
        self.cnt = 0
        self.last_loss = 0
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.rew_norm = reward_normalization
        self.n_step = estimation_step

        self.total_bandwidth = total_bandwidth
        self.loss_dict = {}
        
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optim, start_factor=1.0, end_factor=0.2, total_iters=50
        )

    # === use to compute return ===

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        act_next = self.forward(batch, model="target_actor", input="obs_next").act
        # print(batch)
        obs_next = to_torch(batch.obs_next,device=self.device, dtype=torch.float32)
        noise = torch.randn(size=act_next.shape, device=act_next.device) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        act_next += noise
        act_next = F.softmax(act_next, dim=-1)
        target_q = self.target_critic.q_min(obs_next, act_next)
        return target_q

    #   === update logic ===
    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(batch, buffer, indices, self._target_q, self.gamma, self.n_step, self.rew_norm)

    def update_critic(self, batch: Batch):
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)
        act = to_torch(batch.act, device=self.device, dtype=torch.long)
        current_q1, current_q2 = self.critic(obs, act)
        current_q1 = current_q1.flatten()
        current_q2 = current_q2.flatten()
        target_q = batch.returns.flatten()
        td1 = current_q1 - target_q
        td2 = current_q2 - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        q1_loss = (td1.pow(2) * weight).mean()
        q2_loss = (td2.pow(2) * weight).mean()
        Q_loss = q1_loss + q2_loss
        self.critic_optim.zero_grad()
        Q_loss.backward()
        self.critic_optim.step()
        return td1, td2, q1_loss, q2_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # update double critic
        td1, td2, q1_loss, q2_loss = self.update_critic(batch)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self.cnt % self.update_actor_freq == 0:
            obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)
            act = self.forward(batch).act

            # actor_loss = self.actor.loss(act, obs)-self.critic.q_min(obs, act).mean()
            actor_loss = -self.critic.q_min(obs, act).mean()
            # total_loss, self.loss_dict = self.actor.compute_loss(actor_loss)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.last_loss = actor_loss.item()
            self.actor_optim.step()

            # soft update
            self.soft_update(self.target_critic, self.critic, self.tau)
            self.soft_update(self.target_actor, self.actor, self.tau)

            print('q1 /q2 all_loss:', q1_loss, q2_loss, actor_loss)
            # for key, value in self.loss_dict.items():
            #     print(f"{key}: {value}")
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean():.4f}!")
                else:
                    print(f"❌ No gradient for {name}!")
        self.cnt += 1

        return {
            "loss/critic1": q1_loss.item(),
            "loss/critic2": q2_loss.item(),
            **self.loss_dict,
            "actor_loss": self.last_loss,
        }

    # process_fn -> learn -> post_process_fn(in base.ignore)
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)  # get nstep return
        result = self.learn(batch, **kwargs)  # log data
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result

    # === forward logic ===
    # μ(s).action is continuous.
    def forward(self, batch: Batch, model: str = "actor", input: str = "obs", exploration_fn=None, **kwargs: Any,) -> Batch:
        model = self.actor if model == "actor" else self.target_actor
        obs = to_torch(batch[input], device=self.device, dtype=torch.float32)
        # probs, allocs, actions = model(obs, exploration_fn)
        actions = model(obs)
        hidden = None
        # print(actions)
      
        return Batch(act=actions, state=hidden)

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self.explore_noise is None:
            return act
        if isinstance(act, np.ndarray):
            noise = np.random.normal(loc=0, scale=self.explore_noise, size=act.shape)
            act = act + noise
            act = self._softmax(act)
        return act

    def _softmax(self, act: np.ndarray):
        act_tensor = F.softmax(torch.tensor(act, dtype=torch.float32), dim=-1)
        return act_tensor.cpu().numpy()
