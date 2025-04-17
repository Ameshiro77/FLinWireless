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


class SubAllocEnv(gym.Env):
    def __init__(self, args, dataset: FedDataset, clients: list[Client], model=None, env_type="train"):
        self.args = args
        self.dataset = dataset
        # 用于决定训练RL时是否需要真正训练。
        self.env_type = env_type
        if env_type == "train":
            self.is_fed_train = args.fed_train
        else:
            self.is_fed_train = True
        self.is_alloc_train = False  # 用于区分是否在训练。训练时需要针对选中客户端做cost归一化，奖励函数不一样。
        self.comm_rounds = 0
        self.alloc_steps = args.alloc_steps
        self.clients = clients
        self.selects_num = args.top_k
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.selects_num,), dtype=np.float32
        )
        self.selects = [0] * self.selects_num  # index
        self.max_T, self.min_T, self.max_E, self.min_E = 0, 0, 0, 0
        # STATE
        self.gains = [client.attr_dict["gain"] for client in self.clients[:5]]
        self.data_sizes = [len(client.local_dataset) for client in self.clients[:5]]
        self.frequecies = [client.attr_dict["cpu_frequency"] for client in self.clients[:5]]
        self.distances = [client.attr_dict["distance"] for client in self.clients[:5]]
        self.init_allocations = np.ones(self.selects_num) / self.selects_num
        self.init_times = [self.min_T] * self.selects_num
        self.init_energies = [self.min_E] * self.selects_num
        self.all_allocations = np.zeros(self.selects_num)
        self.all_times = np.zeros(self.selects_num)
        self.all_energies = np.zeros(self.selects_num)
        self.observation_space = spaces.Dict(
            {
                "gains": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.selects_num,),
                    dtype=np.float32,
                ),
                "data_sizes": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.selects_num,),
                    dtype=np.float32,
                ),

                "frequecies": spaces.Box(
                    low=-100, high=100, shape=(self.selects_num,), dtype=np.float32
                ),
                "allocations": spaces.Box(
                    low=0, high=1.0, shape=(self.selects_num,), dtype=np.float32
                ),
                "times": spaces.Box(
                    low=-100, high=100, shape=(self.selects_num,), dtype=np.float32
                ),
                "energys": spaces.Box(
                    low=-100, high=100, shape=(self.selects_num,), dtype=np.float32
                ),
            }
        )
        self.total_bandwidth = TOTAL_BLOCKS
        self.observation = {}

    # get min-max T&E of selected clients for stationary use
    # but in fed env,we use the whole T&E
    def _get_boundary_cost(self, selects):
        # selected_indices = [i for i, mask in enumerate(selects_mask) if mask == 1]
        selected_clients = [self.clients[i] for i in selects]
        min_rb_lst = []
        max_rb_lst = []
        for client in selected_clients:  # 得到最大/小rate时的rb。（实验结果递增）
            min_rate = max_rate = 0
            min_rb = max_rb = 1
            for i in range(TOTAL_BLOCKS):
                client.set_rb_num(i + 1)
                rate = client.get_transmission_rate()
                if rate > max_rate:
                    max_rb = i + 1
                if rate < min_rate:
                    min_rb = i + 1
            min_rb_lst.append(min_rb)
            max_rb_lst.append(max_rb)
        # 算max min T
        max_T, min_T, max_E, min_E = 0, 0, 0, 0
        for client in selected_clients:  # 得到最大/小rate时的rb。（实验结果递增）
            client.set_rb_num(max_rb)  # T min (rate max)
        all_T, all_E = zip(*[client.get_cost() for client in selected_clients])
        min_T, min_E = min(all_T), min(all_E)
        for client in selected_clients:
            client.set_rb_num(int(TOTAL_BLOCKS/len(self.clients)))  # T min (rate max)
        all_T, all_E = zip(*[client.get_cost() for client in selected_clients])
        max_T, max_E = max(all_T), sum(all_E)

        print(f" * Clients: max_T:{max_T}, min_T:{min_T}, max_E:{max_E}, min_E:{min_E}")
        return max_T, min_T, max_E, min_E

    def set_selects(self, selects):
        self.selects = sorted(selects)
        self.max_T, self.min_T, self.max_E, self.min_E = self._get_boundary_cost(self.selects)
        self.data_sizes = [len(self.clients[i].local_dataset) for i in self.selects]
        self.gains = [self.clients[i].attr_dict["gain"] for i in self.selects]
        self.frequecies = [self.clients[i].attr_dict["cpu_frequency"] for i in self.selects]
        self.distances = [self.clients[i].attr_dict["distance"] for i in self.selects]

    # 需要返回初始state和info。
    # must select before reset and step.
    def reset(self, seed=42, options=None):

        # 重置各自客户端模型 以及 全局模型
        blank = np.zeros(self.selects_num)
        info = {}  # must dict
        obs = {
            "gains": self.gains,
            "data_sizes": self.data_sizes,
            "frequecies": self.frequecies,
            # "allocations": self.init_allocations,
            # "times": [self.min_T] * self.selects_num,
            # "energys": [self.min_E] * self.selects_num,
            "all_allocations": blank,
            "all_times": blank,
            "all_energies": blank,
        }
        self.all_allocations = blank
        self.all_energies = blank
        self.all_times = blank
        self.comm_rounds = 0
        self.observation.update(obs)
        obs = self.dict_to_vector(obs)
        print("select:", self.selects)
        print("reset", obs.reshape(6, -1))
        return obs, info

    """
    alloc bandwidth.
    action : a probs distribution of N selects.
    must select before step
    """

    def step(self, action):
        print("\n==Allocs==")
        print(f"train or test:{self.env_type}")
        print(action)
        if (action.sum()-1) > 0.001:
            raise ValueError('bandwidth sum error')
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if np.any(action < 0):
            raise ValueError('action must be >= 0 ')
        args = self.args
        self.comm_rounds += 1
        print("comm_rounds:", self.comm_rounds)
        alloc_blocks = np.floor(action * self.total_bandwidth).astype(np.int32)
        print(alloc_blocks)
        # === 模拟通信 ==
        stats = []
        select_clients = [self.clients[i] for i in self.selects]

        # ==================
        # 匈牙利匹配
        cost_matrix = np.zeros((self.selects_num, self.selects_num))
        for i, client in enumerate(select_clients):
            for j, rb_num in enumerate(alloc_blocks):
                client.set_rb_num(max(1, rb_num))  # 确保最小资源为1
                T, E = client.get_cost()
                time_rew = (T - self.min_T) / (self.max_T - self.min_T)
                energy_rew = (E - self.min_E) / (self.max_E - self.min_E)
                reward = (- args.rew_b * time_rew - args.rew_c * energy_rew) * 20
                cost_matrix[i][j] = -reward
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_alloc = np.zeros(self.selects_num, dtype=np.int32)
        for i, j in zip(row_ind, col_ind):
            best_alloc[i] = max(1, alloc_blocks[j])  # 确保每个 client 至少一个
        alloc_blocks = best_alloc
        stats = []
        # ====================

        # ======================
        # 获得当前的T和E:
        for client, rb_num in zip(select_clients, alloc_blocks):
            if rb_num > 0:  # 分配了资源块就算cost
                client.set_rb_num(rb_num)
            else:
                client.set_rb_num(1)  # 如果没有分配就当最低的算
            T, E = client.get_cost()
            stat = {"time": T, "energy": E}
            stats.append(stat)

        times = [stat["time"] for stat in stats]
        energys = [stat["energy"] for stat in stats]
        total_time = max(times)
        total_energy = sum(energys)
        # ======================

        # == 设计奖励 & 返回值 ==
        time_rew = (total_time - self.min_T) / (
            self.max_T - self.min_T
        )  # min-max normalize
        energy_rew = (total_energy - self.min_E) / (self.max_E - self.min_E)
        reward = (- args.rew_b * time_rew - args.rew_c * energy_rew) * 100
        # min_rew = (- args.rew_b - args.rew_c) * 100
        # reward = min_rew if reward < min_rew else reward
        print(
            f"Bandwidth alloc reward: {reward} total_time: {total_time} total_energy: {total_energy} reward T&E: {-time_rew}, {-energy_rew}")
        # ======================

        max_all_T = self.max_T * self.alloc_steps
        max_all_E = self.max_E * self.alloc_steps
        self.all_allocations = (self.all_allocations + alloc_blocks / self.total_bandwidth)
        self.all_times = self.all_times + times
        self.all_energies = self.all_energies + energys

        # step函数返回：observation, reward, done, done ,info
        # only update dynamic here.must select before step!
        # 如果除以总轮数 反映相对总值 如果除以当前轮数 相当于归一化到0~1
        obs = {
            "gains": self.gains,
            "data_sizes": self.data_sizes,
            "frequecies": self.frequecies,
            # "allocations": action,
            # "times": [(stat["time"]-self.min_T)/(self.max_T-self.min_T) for stat in stats],
            # "energys": [(stat["energy"]-self.min_E)/(self.max_E-self.min_E) for stat in stats]
            "all_allocations": self.all_allocations / self.comm_rounds,
            "all_times": minmax_normalize(self.all_times, 0),
            "all_energies": minmax_normalize(self.all_energies, 0)
        }
        print(obs)
        self.observation.update(obs)
        observasion = self.dict_to_vector(self.observation)
        done = self.comm_rounds >= self.alloc_steps
        info = {
            "communication_rounds": self.comm_rounds,
            "total_time": total_time,
            "total_energy": total_energy,
            "sub_reward": reward,
        }
        print(observasion.reshape((6, -1)))
        return observasion, reward, done, done, info  # 新gym返回五个值。

    def dict_to_vector(self, observation):  # tianshou not support dict observation
        vector = []
        for key, value in observation.items():
            if isinstance(value, (int, float)):
                # 单个值直接加入
                vector.append(value)
            elif isinstance(value, (list, np.ndarray)):
                # 对列表或数组进行标准化。对于某些特殊值特殊处理
                if key == "allocations" or key.startswith("all_"):
                    value = np.array(value, dtype=np.float32)
                    vector.extend(value)
                else:
                    value = np.array(value, dtype=np.float32)
                    if len(value) > 0:  # 确保列表不为空
                        value_mean = np.mean(value)
                        value_std = np.std(value)
                        if value_std != 0:  # 防止除以零
                            value_normalized = (value - value_mean) / value_std
                        else:
                            value_normalized = value  # 如果标准差为零，标准化后仍为原值
                        vector.extend(value_normalized)
                    else:
                        vector.extend(value)  # 空列表直接加入
        return np.array(vector, dtype=np.float32)

    def close(self):
        print("federated progress done...")


def make_sub_env(args, dataset, clients, model):
    args = get_args()
    model = MNISTResNet()
    attr_dicts = init_attr_dicts(args.num_clients)
    dataset = FedDataset(args)
    # == 初始化客户端数据集
    clients = init_clients(args.num_clients, model, dataset, attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
    # ==

    env = SubAllocEnv(args, dataset, clients, model, env_type="train")
    train_envs = SubAllocEnv(args, dataset, clients, model, env_type="train")
    test_envs = SubAllocEnv(args, dataset, clients, model, env_type="test")

    train_envs = CustomDummyVectorEnv([lambda: env for _ in range(args.training_num)])
    test_envs = CustomDummyVectorEnv([lambda: env for _ in range(args.test_num)])

    return env, train_envs, test_envs


if __name__ == "__main__":
    args = get_args()
    args = get_args()
    dataset = FedDataset(args)
    model = MNISTResNet()
    attr_dicts = init_attr_dicts(args.num_clients)
    # == 初始化客户端数据集
    clients = init_clients(args.num_clients, model, dataset, attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
    # ==
    # === 计算 state_dim action_dim ===

    action_dim = 5

    env, train_envs, test_envs = make_sub_env(args, dataset, clients, model)

    state_dim = 0
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            state_dim += 1  # Discrete 空间是标量
        elif isinstance(space, spaces.Box):
            state_dim += np.prod(space.shape)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    # == actor
    from net.alloctor import *
    actor = DirichletPolicy(state_dim=state_dim, action_dim=5, hidden_dim=256, constant=10).cuda()
    # actor = MLPAllocator(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    # actor = DiffusionAllocor(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 1e-3)

    # === 指定 critic 以及对应optim ===
    critic = Critic_V(state_dim=state_dim,).to(args.device)
    #critic = DoubleCritic(state_dim=state_dim, action_dim=5, hidden_dim=256).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # policy = TD3Policy(actor,
    #                    actor_optim,
    #                    critic,
    #                    critic_optim=critic_optim,
    #                    device=args.device,
    #                    tau=0.005,
    #                    gamma=0.99,
    #                    policy_noise=0.2,
    #                    noise_clip=0.5,
    #                    alpha=2.5,
    #                    )

    policy = PPOPolicy(
        actor,
        critic,
        optim_lr=1e-4,
        dist_fn=torch.distributions.Categorical,
        device=args.device,
        ent_coef=0.02,
    )

    # )

    # === tianshou 相关配置 ===
    import random
    buffer = VectorReplayBuffer(512, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buffer=buffer)
    selects = random.sample(range(10), 5)
    print("=====")
    train_envs.set_selects(selects)
    train_envs.reset()

    writer = SummaryWriter("./debug")
    logger = BasicLogger(writer)  # 创建基础日志记录器

    # trainer = OffpolicyTrainer(
    #     policy,
    #     train_collector,
    #     test_collector,
    #     max_epoch=args.epochs,
    #     step_per_epoch=20,
    #     step_per_collect=20,
    #     episode_per_test=args.test_num,
    #     batch_size=128,
    #     update_per_step=1,
    #     logger=logger,
    #     save_best_fn=None,
    # )
    # train_collector.collect(20)
    # exit()
    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        repeat_per_collect=2,  # diff with offpolicy_trainer
        step_per_epoch=20,
        step_per_collect=20,
        episode_per_test=args.test_num,
        batch_size=16,
        logger=logger,
        save_best_fn=None
    )

    print("start!!!!!!!!!!!!!!!")
    for epoch, epoch_stat, info in trainer:
        # selects = random.sample(range(10), 5)
        # train_envs.set_selects(selects)
        print("=======\nEpoch:", epoch)
        print(epoch_stat)
        print(info)
        logger.write('epochs/stat', epoch, epoch_stat)

    print("done")
