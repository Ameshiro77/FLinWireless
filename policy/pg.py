from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd


class PGPolicy(BasePolicy):
    """
    model: s->logits
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = True,
        action_scaling: bool = False,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        lr_scheduler: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs,
        )
        self.actor = model
        try:
            if action_scaling and not np.isclose(model.max_action, 1.0):  # type: ignore
                import warnings

                warnings.warn(
                    "action_scaling and action_bound_method are only intended"
                    "to deal with unbounded model action space, but find actor model"
                    f"bound action space with max_action={model.max_action}."
                    "Consider using unbounded=True option of the actor model,"
                    'or set action_scaling to False and action_bound_method to "".'
                )
        except Exception:
            pass
        self.optim = optim
        self.dist_fn = dist_fn
        self.device = device
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        lr_maxt = 100
        if self.lr_scheduler:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(
                self.ret_rms.var + self._eps
            )
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any,) -> Batch:
        obs = to_torch(batch.obs, device=self.device, dtype=torch.float32)  # bs*(numclients x statedim)
        print("is train:", self.training)
        act, log_ll, entropy = self.actor(obs, self.training)
        return Batch(act=act, state=None, log_prob=log_ll, entropy=entropy)

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                ret = to_torch(minibatch.returns, torch.float, self.device)
                log_prob = result.log_prob.reshape(len(ret), -1).transpose(0, 1)
                print(ret, log_prob)
                loss = -(log_prob * ret).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())

            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    print(f"✅ Gradient exists for {name}, mean grad: {param.grad.mean()}!")

        return {"loss": losses}

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result
