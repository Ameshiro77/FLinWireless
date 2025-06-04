import torch
from tianshou.env import DummyVectorEnv
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa
import gymnasium as gym
from gym import spaces
import torch.nn.functional as F
from env import *
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from policy import *
from tools import *
from net import *
from tools.misc import *

from datetime import datetime
from SubAllocEnv import *


class AllocActor2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.DiffAllocator = DiffusionAllocor(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).cuda()
        self.MLPAllocator = MLPAllocator(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.LSTMProcessor = ObsProcessor(window_size=5, hidden_size=10)

    def forward(self, obs_dict, is_training=True):
        x = self.LSTMProcessor(obs_dict)  # B N 13
        x = x.view(x.shape[0], -1)  # B * N * dim ->B * (N * Input)
        x = self.MLPAllocator(x)
        # print(x.shape)
        # exit()
        return x


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    args.num_clients = 10
    args.num_choose = 10
    action_dim = 10
    env, train_envs, test_envs = make_sub_env(args)

    state_dim = 0
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            state_dim += 1  # Discrete 空间是标量
        elif isinstance(space, spaces.Box):
            state_dim += np.prod(space.shape)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    state_dim = 130
    # == actor
    from net.alloctor import *

    actor = AllocActor2(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), 1e-4)

    # === 指定 critic 以及对应optim ===

    critic = Critic_Q(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    policy = DDPGPolicy(
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        device=args.device,
        gamma=0.99,
        exploration_noise=None,
    ).to(args.device)

    # === tianshou 相关配置 ===
    import random
    buffer = VectorReplayBuffer(512, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buffer=buffer)

    writer = SummaryWriter("./debug")
    logger = BasicLogger(writer)  # 创建基础日志记录器

    trainer = OffpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        step_per_epoch=20,
        step_per_collect=20,
        episode_per_test=args.test_num,
        batch_size=128,
        update_per_step=1,
        logger=logger,
        save_best_fn=None,
    )

    print("start!!!!!!!!!!!!!!!")
    for epoch, epoch_stat, info in trainer:
        print("=======\nEpoch:", epoch)
        print(epoch_stat)
        print(info)
        logger.write('epochs/stat', epoch, epoch_stat)

    print("done")
