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
from tianshou.exploration import BaseNoise, GaussianNoise
import numpy as np
from tianshou.data import Batch, to_torch_as


class DDPGPolicy(BasePolicy):
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
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        alpha: float = 2.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        # about tasks:
        total_bandwidth: int = 100,
        num_choose: int = 5,
        task='fed',
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

    # === use to compute return ===

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        act_next = self.forward(batch, model="target_actor", input="obs_next").act
        obs_next = torch.FloatTensor(batch.obs_next).to(self.device)
        target_q = self.target_critic(obs_next, act_next)
        return target_q

    #   === update logic ===
    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(batch, buffer, indices, self._target_q, self.gamma, self.n_step, self.rew_norm)

    def update_critic(self, batch: Batch):
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)
        act = to_torch(batch.act, device=self.device, dtype=torch.long)
    
        current_q = self.critic(obs, act)
        current_q = current_q.flatten()
        target_q = batch.returns.flatten()
        td1 = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        q1_loss = (td1.pow(2) * weight).mean()
        self.critic_optim.zero_grad()
        q1_loss.backward()
        self.critic_optim.step()
        return td1, None, q1_loss, None

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # update double critic  only use one Q!
        td1, td2, q1_loss, q2_loss = self.update_critic(batch)
        batch.weight = td1

        # actor
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)
        actor_loss = -self.critic(obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        for name, param in self.actor.named_parameters():
            print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean()}!")
        self.soft_update(self.target_critic, self.critic, self.tau)
        self.soft_update(self.target_actor, self.actor, self.tau)
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": q1_loss.item(),
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
    def forward(self, batch: Batch, model: str = "actor", input: str = "obs", **kwargs: Any,) -> Batch:
        model = self.actor if model == "actor" else self.target_actor
        obs = to_torch(batch[input], device=self.device, dtype=torch.float32)
        actions, hidden = model(obs), None
        actions = F.softmax(actions, dim=-1)
        return Batch(act=actions, state=hidden)

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self.explore_noise is None:
            return act
        if isinstance(act, np.ndarray):
            act = act + self.explore_noise(act.shape)
            act_tensor = F.softmax(torch.tensor(act, dtype=torch.float32), dim=-1)
            return act_tensor.cpu().numpy()
        return act
