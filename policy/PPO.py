from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_torch
from tianshou.policy import BasePolicy, A2CPolicy
from tianshou.utils.net.common import ActorCritic
from torch.optim.lr_scheduler import CosineAnnealingLR


class PPOPolicy(A2CPolicy):

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: Optional[torch.optim.Optimizer],
        critic: torch.nn.Module,
        critic_optim: Optional[torch.optim.Optimizer],
        # optim_lr: float,
        dist_fn: Type[torch.distributions.Distribution],
        device: Optional[Union[str, torch.device]] = None,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        reward_normalization: bool = True,
        value_clip: bool = False,
        ent_coef: float = 0.01,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        scheduler_iters: int = 100,
        optim=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self.device = device
        self.dist_fn = dist_fn
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._weight_ent = ent_coef
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._grad_norm = max_grad_norm
        self._batch = max_batchsize
        self._lambda = gae_lambda
        self.step_times = 0
        # self._actor_critic: ActorCritic
        # self.optim = torch.optim.Adam(self._actor_critic.parameters(), lr=optim_lr)
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optim, start_factor=1.0, end_factor=0.05, total_iters=scheduler_iters
        ) if scheduler_iters > 0 else None
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.actor_optim, T_max=scheduler_iters, eta_min=1e-5
        # ) if scheduler_iters > 0 else None

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch).log_prob
        print("ooooooold", batch.logp_old)
        return batch

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(to_torch(minibatch.obs, device=self.device, dtype=torch.float32)))
                v_s_.append(self.critic(to_torch(minibatch.obs_next, device=self.device, dtype=torch.float32)))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any,) -> Batch:
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)  # bs*(numclients x statedim)
        # print(obs)
        print("is train:", self.training)
        act, log_ll, entropy = self.actor(obs, self.training)
        return Batch(act=act, state=None, log_prob=log_ll, entropy=entropy)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                result = self(minibatch)
                log_prob = result.log_prob
                # print(log_prob)
                # exit()
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                # print("PROBS", log_prob, minibatch, minibatch.logp_old)
                # exit()
                # print(log_prob.shape, minibatch.logp_old.shape)
                # exit()
                clipped_delta_logp = torch.clamp(log_prob - minibatch.logp_old, -1.1, 1.1)
                ratio = torch.exp(torch.sum(clipped_delta_logp, dim=-1))  # log(x/y) = logx - logy
                # ratio = (log_prob - minibatch.logp_old).exp().float()  # trick: e^(logx-logy) = x/y cause y is small
                # print("log_prob", log_prob)
                # print("logp_old", minibatch.logp_old)
                # print("adv", minibatch.adv)
                # print("ratio", ratio)
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # print(minibatch)
                # print("surr:", surr1, surr2)
                # calculate loss for critic
                value = self.critic(to_torch(minibatch.obs, device=self.device, dtype=torch.float32)).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = result.entropy.mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                # =============== gradient update ===============
                # self.optim.zero_grad()
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                # print(clip_loss, vf_loss, ent_loss)
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=self._grad_norm
                    )
                self.actor_optim.step()
                self.critic_optim.step()
                #
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
            for name, param in self.actor.named_parameters():
                if param.grad is None:
                    print(f"⚠️ WARNING: No gradient for {name}")
                else:
                    print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean()}")
            # exit()

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        # ON_POLICY TRAINER里，sample_size=0 -> 全取
        batch, indices = buffer.sample(sample_size)

        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)

        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.step_times += 1
        self.updating = False
        self._weight_ent *= 0.95 #熵系数衰减
        print("step_times", self.step_times)
        

        return result
