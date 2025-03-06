import sys  # noqa
import os  # noqa

current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa

import gym
import torch
from gym import spaces
from env.dataset import FedDataset
from env.client import *
from env.models import *
from env.FedEnv import FederatedEnv, make_env
import torch.nn.functional as F
from policy import *
import argparse
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from policy.diffpolicy import DiffusionSAC
from net.model import *
from net.diffusion import *
from logger import InfoLogger

"""
用来测试gym环境 以看算法对不对。
"""

if __name__ == "__main__":
    
    env = gym.make("CartPole-v1")
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])  # 10 个并行环境
    test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])  # 2 个并行环境


    state_dim = env.observation_space.shape
    action_dim = 1

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

    # === 初始化RL模型 ===
    # policy = DiffusionSAC(actor,
    #                       actor_optim,
    #                       args.num_clients,
    #                       critic,
    #                       critic_optim,
    #                       dist_fn=torch.distributions.Categorical,
    #                       device=args.device,
    #                       gamma=0.95,
    #                       estimation_step=3,
    #                       is_not_alloc=args.no_allocation,
    #                       )
    # policy = DiffusionTD3(actor,
    #                       actor_optim,
    #                       critic=critic,
    #                       critic_optim=torch.optim.Adam(critic.parameters(), lr=1e-3),
    #                       device=args.device,
    #                       tau=0.005,
    #                       gamma=0.99,
    #                       training_noise=0.1,
    #                       policy_noise=0.2,
    #                       noise_clip=0.5
    #                       )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=100,
            step_per_epoch=100,
            step_per_collect=100,
            episode_per_test=1,
            batch_size=64,
            update_per_step=0.2,
            logger=None,
        )
    
    for i in range(10):  # 训练总数
        train_collector.collect(n_step=2)
        print("===========")
        # 使用采样出的数据组进行策略训练
        losses = policy.update(1, train_collector.buffer)
