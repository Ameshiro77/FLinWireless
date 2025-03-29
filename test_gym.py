import sys  # noqa
import os  # noqa

current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa

import torch


import torch.nn.functional as F
import argparse
from tianshou.policy import DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger

from env import *
from policy import *
from net import *

from datetime import datetime
import gymnasium as gym
from tianshou.env import DummyVectorEnv

"""
用来测试gym环境 以看算法对不对。
"""

if __name__ == "__main__":

    env = gym.make("Pendulum-v1")
    train_envs = DummyVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(1)]) 
    test_envs = DummyVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(1)]) 

    args = get_args()
    args.no_logger = True
    args.task = 'gym'
    # === 计算 state_dim action_dim ===
    state_dim = 3
    action_dim = 1
 
    # === 指定 actor 以及对应optim ===
    actor, critic = choose_actor_critic(state_dim, action_dim, args)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化RL模型 ===
    policy = choose_policy(actor, actor_optim, critic, critic_optim, args)

    # === tianshou 相关配置 ===
    buffer = VectorReplayBuffer(512, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer)
    test_collector = Collector(policy, test_envs, buffer=buffer)

    # for i in range(5):  # 训练总数
    #     train_collector.collect(n_step=2)
    #     losses = policy.update(1, train_collector.buffer)
    #     print(losses)
    # exit()


    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=100,
        step_per_epoch=100,
        step_per_collect=100,
        episode_per_test=1,
        batch_size=64,
        update_per_step=1,
    )

    # === eval ===
    print("======== evaluate =======")
    np.random.seed(args.seed)

    policy.eval()
    env = DummyVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(1)]) 
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
