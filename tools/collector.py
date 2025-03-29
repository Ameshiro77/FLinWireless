import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
import torch
from tianshou.data.collector import Collector
from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)


class CustomCollector(Collector):
    def __init__(self, policy, env, buffer=None, preprocess_fn=None, exploration_noise=False):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def reset(self, reset_buffer: bool = True, gym_reset_kwargs: Optional[Dict[str, Any]] = None,) -> None:
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            probs={},
            allocs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def collect(self, n_step=None, n_episode=None, random=False, render=None, no_grad=True, gym_reset_kwargs=None):
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if no_grad:
                with torch.no_grad():  # faster than retain_grad version
                    if self.exploration_noise:
                        exploration_fn = self.policy.exploration_noise
                    result = self.policy(self.data, last_state, exploration_fn=exploration_fn)
            else:
                if self.exploration_noise:
                    exploration_fn = self.policy.exploration_noise
                result = self.policy(self.data, last_state, exploration_fn=exploration_fn)

            # update state / act / policy into self.data

            # state
            policy = result.get("policy", Batch())
            assert isinstance(policy, Batch)
            state = result.get("state", None)
            if state is not None:
                policy.hidden_state = state  # save state into buffer

            # act    # AND ACT INCLUDES : PROBS / ALLOCS / AND FINAL OUTPUTS
            probs = to_numpy(result.probs)
            allocs = to_numpy(result.allocs)
            act = to_numpy(result.act)
            self.data.update(policy=policy, act=act, probs=probs, allocs=allocs)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )
            done = np.logical_or(terminated, truncated)

            # update else
            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                probs={},
                allocs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }
