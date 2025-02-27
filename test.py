import sys  # noqa
import os  # noqa

current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, 'env'))  # noqa
sys.path.append(os.path.join(current_dir, 'net'))  # noqa
sys.path.append(os.path.join(current_dir, 'policy'))  # noqa

import gym
import torch
from gym import spaces
from env.dataset import FedDataset
from env.client import *
from env.model import *
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

# 设定一些参数
state_dim = 4  # 对于 CartPole 环境，状态空间维度是 4
action_dim = 2  # 对于 CartPole，动作空间维度是 2（左或右）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_bandwidth = 100  # 带宽限制，根据需求调整
top_k = 3  # 选择 K 个客户端

# 定义 Actor 网络
actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=actor_net, max_action=100).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)


# 定义 Critic 网络
class Critic(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.fc2 = torch.nn.Linear(256, 1)

    def forward(self, obs, act:torch.tensor):
        act = act.view(-1,1)
        print(obs,act)
        x = torch.cat([obs, act], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


critic = Critic(state_dim, action_dim).to(device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

# 初始化 DiffusionSAC 策略
policy = DiffusionSAC(actor=actor,
                      actor_optim=actor_optim,
                      action_dim=action_dim,
                      critic=critic,
                      critic_optim=critic_optim,
                      device=device,
                      total_bandwidth=total_bandwidth,
                      alpha=0.2,
                      tau=0.005,
                      gamma=0.99,
                      reward_normalization=False,
                      estimation_step=1,
                      lr_decay=True,
                      lr_maxt=1000,
                      pg_coef=0.5,
                      top_k=top_k)

# 创建 Gym 环境
env = gym.make('CartPole-v1')


# 创建 SubprocVecEnv 用于并行化训练
def make_env():
    env = gym.make('CartPole-v1')
    train_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    return train_envs, test_envs


train_envs, test_envs = make_env()  # 创建 4 个环境进行并行训练

# 定义 ReplayBuffer
buffer = VectorReplayBuffer(64, 1)
train_collector = Collector(policy, train_envs, buffer=buffer)
test_collector = Collector(policy, test_envs, buffer=buffer)
# 设置日志记录器
writer = SummaryWriter()
logger = InfoLogger(writer)  # 创建基础日志记录器

# 定义训练器
result = offpolicy_trainer(policy,
                           train_collector,
                           test_collector,
                           max_epoch=10,
                           step_per_epoch=100,
                           step_per_collect=10,
                           episode_per_test=1,
                           batch_size=10,
                           update_per_step=1,
                           logger=logger)

# 测试训练效果
_,test_env = make_env()  # 创建测试环境
test_collector = Collector(policy, test_env)

# 测试模型性能
test_result = test_collector.collect(n_episode=10)
print(f"Test Result: {test_result}")
