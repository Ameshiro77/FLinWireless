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


def run(env, args):
    state, info = env.reset()
    print("env reset!\ninit state:", state)
    policy = RandomPolicy(env.action_space, args)
    print("start training")
    for _ in range(args.global_rounds):
        action = policy.random_action()
        next_state, reward, done, info = env.step(action)
        print("\nnext_state:", next_state)
        print("reward:", reward)
        if done:
            break
    env.close()


if __name__ == "__main__":

    args = get_args()
    dataset = FedDataset(args)
    model = MNISTResNet()
    optimizer_class = torch.optim.Adam
    attr_dicts = init_attr_dicts(args.num_clients)

    # == 初始化客户端数据集
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
    # ==

    env, train_envs, test_envs = make_env(args, dataset, clients, model=model)

    # run(env, args)

    # ======================
    # 使用tianshou进行训练

    # 计算 state_shape
    state_dim = 0

    if isinstance(env.observation_space, spaces.Dict):
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                state_dim += 1  # Discrete 空间是标量
            elif isinstance(space, spaces.Box):
                state_dim += np.prod(space.shape)
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")
    else:
        state_dim = env.observation_space.shape

    # state_dim = 4
    action_dim = args.num_clients
    print("state_dim:", state_dim, "action_dim:", action_dim)
    # === 指定 actor 以及对应optim ===

    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
                    # activation='Relu'
                    )
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
    policy = DiffusionTD3(actor,
                          actor_optim,
                          critic=critic,
                          critic_optim=torch.optim.Adam(critic.parameters(), lr=1e-3),
                          device=args.device,
                          tau=0.005,
                          gamma=0.99,
                          training_noise=0.1,
                          policy_noise=0.2,
                          noise_clip=0.5
                          )
    # train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])  # 10 个并行环境
    # test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])  # 2 个并行环境
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    for i in range(10):  # 训练总数
        train_collector.collect(n_step=2)
        print("===========")
        # 使用采样出的数据组进行策略训练
        losses = policy.update(1, train_collector.buffer)
