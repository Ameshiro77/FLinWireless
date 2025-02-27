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



if __name__ == '__main__':
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
    # ==

    env, train_envs, test_envs = make_env(args, dataset, clients, model=model)

    # run(env, args)

    # ======================
    # 使用tianshou进行训练

    # 计算 state_shape
    state_dim = 0
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            state_dim += 1  # Discrete 空间是标量
        elif isinstance(space, spaces.Box):
            state_dim += np.prod(space.shape)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    action_dim = 10

    # === 指定 actor 以及对应optim ===
    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=100,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)

    # === 指定 critic 以及对应optim ===
    # critic = DoubleCritic(
    #     state_dim=state_shape,
    #     action_dim=10,
    #     hidden_dim=256
    # ).to(args.device)
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    class Critic(torch.nn.Module):

        def __init__(self, state_dim, action_dim):
            super(Critic, self).__init__()
            self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)  # 输入为状态和动作的拼接
            self.fc2 = torch.nn.Linear(256, 1)

        def forward(self, obs, act):
            # 将状态和动作拼接
            x = torch.cat([obs, act], dim=-1)  # 假设状态和动作在最后一维拼接
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    critic = Critic(state_dim, action_dim).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化RL模型 ===
    policy = DiffusionSAC(actor, actor_optim, 10, critic, critic_optim, args.device, TOTAL_BLOCKS)
    PATH = args.algo + '_ckpt.pth'
    def save_best_fn(model):
        torch.save(model.state_dict(), PATH)
        
    if args.resume == True or args.evaluate == True:
        print("model path:",PATH)
        policy.load_state_dict(torch.load(PATH))
        print("model loaded!")

    
    # tianshou 相关配置
    buffer = VectorReplayBuffer(64, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer)
    test_collector = Collector(policy, test_envs, buffer=buffer)


    if not args.evaluate:
        from datetime import datetime
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f'logs/exp/{timestamp}')
        logger = InfoLogger(writer)  # 创建基础日志记录器
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=args.epochs,
            step_per_epoch=args.global_rounds,
            step_per_collect=args.global_rounds,
            episode_per_test=args.test_num,
            batch_size=args.datas_per_update,
            update_per_step=args.update_per_step,
            logger=logger,
            save_best_fn = save_best_fn
        )

    np.random.seed(args.seed)
    env, _, test_env = make_env(args, dataset, clients, model=model)
    policy.eval()
    collector = Collector(policy, test_env)
    result = collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
