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


def minmax_normalize(x: np.ndarray | list, min_val=None, max_val=None):
    x = np.array(x) if isinstance(x, list) else x
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)


def zscore_normalize(x: np.ndarray | list, mean=None, std=None):
    x = np.array(x) if isinstance(x, list) else x
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    if std == 0:
        return x  # 避免除以零
    return (x - mean) / std


def sum_normalize(x: np.ndarray | list):
    x = np.array(x) if isinstance(x, list) else x
    return x / x.sum()


class SubAllocEnv(gym.Env):
    def __init__(self, args, dataset: FedDataset, clients: list[Client], model=None, env_type="train"):
        self.args = args
        self.dataset = dataset
        # 用于决定训练RL时是否需要真正训练。
        self.env_type = env_type
        self.is_alloc_train = False  # 用于区分是否在训练。训练时需要针对选中客户端做cost归一化，奖励函数不一样。
        self.comm_rounds = 0
        self.alloc_steps = args.alloc_steps
        self.clients = clients

        self.clients_num = len(self.clients)
        self.selects_num = len(self.clients)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.selects_num,), dtype=np.float32
        )
        self.selects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.max_T, self.min_T, self.max_E, self.min_E = self._get_boundary_cost()
        # STATE
        self.gains = np.array([client.attr_dict["gain"] for client in self.clients[:self.selects_num]])
        self.data_sizes = np.array([len(client.local_dataset) for client in self.clients[:self.selects_num]])
        self.frequecies = np.array([client.attr_dict["cpu_frequency"] for client in self.clients[:self.selects_num]])
        self.powers = np.array([client.attr_dict["transmit_power"] for client in self.clients[:self.selects_num]])
        # self.distances = np.array([client.attr_dict["distance"] for client in self.clients[:self.selects_num]])
        self.init_allocations = np.ones(self.selects_num) / self.selects_num
        self.init_times = [self.min_T] * self.selects_num
        self.init_energies = [self.min_E] * self.selects_num

        self.all_allocations = np.zeros(self.selects_num)
        self.all_times = np.zeros(self.selects_num)
        self.all_energies = np.zeros(self.selects_num)

        self.history_allocs = np.zeros((self.clients_num, 5))
        self.history_times = np.zeros(self.clients_num)
        self.history_energies = np.zeros(self.clients_num)

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
    def _get_boundary_cost(self):
        # 仅仅作为近似的手段，主要用于归一化。
        min_rb_lst = []
        max_rb_lst = []
        for client in self.clients:  # 得到最大/小rate时的rb
            min_rate = max_rate = 0
            min_rb = max_rb = 1
            for i in range(TOTAL_BLOCKS):
                client.set_rb_num(i + 1)
                rate = client.get_transmission_rate()
                if rate > max_rate:
                    max_rb = i + 1
                if rate < min_rate:
                    min_rb = i + 1
            min_rb = max(min_rb, TOTAL_BLOCKS/20)  # 过小会导致时间过大，无法正常归一化。xlog(1+1/x)
            min_rb_lst.append(min_rb)
            max_rb_lst.append(max_rb)
        print(min_rb_lst)
        # 算max min T
        min_Ts = []
        max_Ts = []
        for i, client in enumerate(self.clients):  # 得到最大/小rate时的rb。（实验结果递增）
            client.set_rb_num(max_rb_lst[i])  # T min (rate max)
            min_Ts.append(client.get_cost()[0])
            client.set_rb_num(min_rb_lst[i])  # T max
            max_Ts.append(client.get_cost()[0])
        min_T, max_T = min(min_Ts), max(max_Ts)  # min也是近似。

        # 难以求解，故近似。
        for client in self.clients:
            client.set_rb_num(int(TOTAL_BLOCKS/self.clients_num))  # T min (rate max)
        all_E = [client.get_cost()[1] for client in self.clients]
        max_E = max(all_E) * self.selects_num
        min_E = min(all_E) * self.selects_num

        print(f" * Clients: max_T:{max_T}, min_T:{min_T}, max_E:{max_E}, min_E:{min_E}")
        return max_T, min_T, max_E, min_E

    def set_selects(self, selects):
        self.selects = sorted(selects)
        self.max_T, self.min_T, self.max_E, self.min_E = self._get_boundary_cost(self.selects)
        self.data_sizes = [len(self.clients[i].local_dataset) for i in self.selects]
        self.gains = [self.clients[i].attr_dict["gain"] for i in self.selects]
        self.frequecies = [self.clients[i].attr_dict["cpu_frequency"] for i in self.selects]
        # self.distances = [self.clients[i].attr_dict["distance"] for i in self.selects]

    # 需要返回初始state和info。
    # must select before reset and step.
    def reset(self, seed=42, options=None):

        # 重置各自客户端模型 以及 全局模型
        blank = np.zeros(self.selects_num)
        self.history_allocs = np.zeros((self.clients_num, 5))
        info = {}  # must dict
        obs = {
            "gains": zscore_normalize(self.gains),
            "data_sizes": zscore_normalize(self.data_sizes),
            "frequecies": zscore_normalize(self.frequecies),
            "powers": zscore_normalize(self.powers),
            # "gains": self.gains,
            # "data_sizes": self.data_sizes,
            # "frequecies": self.frequecies,
            "allocations": self.history_allocs,
            # "times": blank,
            # "energies": blank,
            # "all_times": blank,
            # "all_energies": blank,
        }
        self.all_allocations = blank
        self.all_energies = blank
        self.all_times = blank
        self.comm_rounds = 0
        self.observation.update(obs)
        # obs = self.dict_to_vector(obs)
        # print("select:", self.selects)
        # print("reset", obs.reshape(6, -1))
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
        # if (action.sum()-1) > 0.001:
        #     raise ValueError('bandwidth sum error')
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
        if args.hungarian:
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
        for client in self.clients:
            client.update_state()

        # == 设计奖励 & 返回值 ==
        time_rew = (total_time - self.min_T) / (
            self.max_T - self.min_T
        )  # min-max normalize
        energy_rew = (total_energy - self.min_E) / (self.max_E - self.min_E)
        # reward = (- args.rew_b * time_rew - args.rew_c * energy_rew) * 100
        reward = (total_time + total_energy / 10) * -10
        if (action.sum()-1) > 0.001:
            reward = -500
        # min_rew = (- args.rew_b - args.rew_c) * 100
        # reward = min_rew if reward < min_rew else reward
        print(
            f"Bandwidth alloc reward: {reward} total_time: {total_time} total_energy: {total_energy} reward T&E: {-time_rew}, {-energy_rew} \n\
                max T AND E:{self.max_T}, {self.max_E} min T AND E:{self.min_T}, {self.min_E}")
        # ======================

        self.all_allocations = (self.all_allocations + alloc_blocks / self.total_bandwidth)
        self.all_times = self.all_times + times
        self.all_energies = self.all_energies + energys

        # step函数返回：observation, reward, done, done ,info
        # only update dynamic here.must select before step!
        # 如果除以总轮数 反映相对总值 如果除以当前轮数 相当于归一化到0~1

        self.history_allocs[:, :-1] = self.history_allocs[:, 1:]  # numpy has no pop()
        self.history_allocs[:, -1] = action
        self.history_times[:-1] = self.history_times[1:]
        self.history_times[-1] = total_time
        self.history_energies[:-1] = self.history_energies[1:]
        self.history_energies[-1] = total_energy

        obs = {
            "gains": zscore_normalize(self.gains),
            "data_sizes": zscore_normalize(self.data_sizes),
            "frequecies": zscore_normalize(self.frequecies),
            "powers": zscore_normalize(self.powers),
            # "gains": self.gains,
            # "data_sizes": self.data_sizes,
            # "frequecies": self.frequecies,
            "allocations": self.history_allocs,
            # "times": self.history_times,
            # "energies": self.history_energies,
            # "all_times": sum_normalize(self.all_times),
            # "all_energies": sum_normalize(self.all_energies)
        }
        # print(obs)
        self.observation.update(obs)
        # observasion = self.dict_to_vector(self.observation)
        # print(observasion.reshape((6, -1)))
        info = {
            "communication_rounds": self.comm_rounds,
            "total_time": total_time,
            "total_energy": total_energy,
            "sub_reward": reward,
        }

        done = self.comm_rounds >= self.alloc_steps
        return obs, reward, done, done, info  # 新gym返回五个值。

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


def make_sub_env(args):
    model = choose_model(args)
    attr_dicts = init_attr_dicts(args.num_clients)
    dataset = FedDataset(args)
    # == 初始化客户端数据集
    clients = init_clients(args.num_clients, model, dataset, attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
    # ==

    env = SubAllocEnv(args, dataset, clients, model, env_type="train")

    train_envs = CustomDummyVectorEnv([lambda: env for _ in range(args.training_num)])
    test_envs = CustomDummyVectorEnv([lambda: env for _ in range(args.test_num)])

    return env, train_envs, test_envs
