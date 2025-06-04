import torch
import time
from tianshou.env import DummyVectorEnv
from client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import argparse  # 添加 argparse 模块
import gymnasium as gym
from gym import spaces
from dataset import FedDataset
import torch.nn.functional as F
from models import *
from client import *
from config import *
from scipy.optimize import linear_sum_assignment


def acc_func(x, dataset):
    if dataset == 'MNIST':
        k = 0.1
        n = 2.3
        a = 0.5
    if dataset == 'Fashion':
        k = 0.35
        n = 2.5
        a = 0.55
    elif dataset == 'CIFAR10':
        k = 1
        n = 3
        a = 0.65
    return 100 * np.tanh((k * x) / (n - (x+a)**2))


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


class FederatedEnv(gym.Env):

    def __init__(self, args, dataset: FedDataset, clients: list[Client], model):
        super().__init__()

        # == log fed progress.irrelevant with RL. ==
        self.task = args.task
        from datetime import datetime
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.log_file = f"./logs/log_{timestamp}.txt"
        self.is_log = args.log_fed
        # ==
        self.original_model = copy.deepcopy(model)  # 每次reset用。

        self.model = model.cuda()  # 全局模型

        self.args = args
        self.dataset = dataset
        self.dataset_name = args.dataset
        # 用于决定训练RL时是否需要真正训练。
        self.is_fed_train = args.fed_train
        if args.evaluate:
            self.is_fed_train = True
        self.is_use_hungarian = args.hungarian
        self.acc_delta = args.acc_delta
        # 初始化客户端以及对应的数据集。
        self.clients = clients
        self.local_rounds = args.local_rounds  # 训练轮数
        self.global_rounds = args.global_rounds
        self.latest_acc = 0

        self.clients_num = len(self.clients)
        self.num_choose = args.num_choose
        self.parti_times = np.zeros(self.clients_num)

        self.latest_global_model = self._get_model_parameters()
        self.current_round = 0
        self.data_sizes = np.array([len(client.local_dataset) for client in self.clients])
        self.gains = zscore_normalize([client.attr_dict["gain"] for client in self.clients])
        self.frequencies = zscore_normalize([client.attr_dict["cpu_frequency"] for client in self.clients])
        self.powers = zscore_normalize([client.attr_dict["transmit_power"] for client in self.clients])
        self.base_noise = args.base_noise
        self.noise_epsilon = 1e-5
        self.total_bandwidth = TOTAL_BLOCKS
        self.max_acc = {
            'MNIST': 0.99,
            'Fashion': 0.95,
            'CIFAR10': 0.44
        }

        # self.qualities = self._get_data_qualities()
        self.qualities = np.zeros(self.clients_num)
        # 归一化用 . min T = min single T
        # rew
        if args.norm_cost:
            self.max_T, self.min_T, self.max_E, self.min_E = self._get_boundary_cost()

        self.data_qualities = np.zeros(self.clients_num)  # reset()里计算
        self.all_allocations = np.zeros(self.clients_num)
        self.all_times = np.zeros(self.clients_num)
        self.all_energies = np.zeros(self.clients_num)

        self.window_size = args.window_size
        self.history_allocs = np.zeros((self.clients_num, self.window_size))
        # STATE ACTION 的SPACE
        self.action_space = spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "data_qualities": spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32),
            "data_sizes": spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32),
            "parti_times": spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32),
            "gains": spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32),
            "alloc": spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32),
        })
        self.observation = {}

    def _get_data_qualities(self):
        losses = np.zeros(self.clients_num)
        train_loss = []
        for id, client in enumerate(self.clients):
            _, stat = client.local_train(is_subset=True, local_rounds=5)  # 先训练本地的一小部分
            train_loss.append(stat['loss'])
            loss = client.valid_train(self.dataset.valid_data)  # 再用testdata训练获得loss来评估
            losses[id] = loss
        # self._reset_clients()
        qualities = losses
        print('qual:', qualities)
        # exit()
        return qualities

    def _reset_clients(self):
        for client in self.clients:
            client.set_model_parameters(self.original_model.state_dict())
            client.set_optim()
            client.reset_state()

    def _get_model_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def _set_model_parameters(self, model_parameters_dict):
        self.model.load_state_dict(model_parameters_dict)
        for client in self.clients:
            client.set_model_parameters(model_parameters_dict)

    def _set_clients_parameters(self, model_parameters_dict):
        for client in self.clients:
            client.set_model_parameters(model_parameters_dict)

    # get min-max T&E. only used in init
    def _get_boundary_cost(self):
        # 仅仅作为近似的手段，主要用于归一化。
        min_rb_lst = []
        max_rb_lst = []
        for client in self.clients:  # 得到最大/小rate时的rb
            min_rate = max_rate = 0
            min_rb = max_rb = 1
            max_rb = TOTAL_BLOCKS
            min_rb = max(min_rb, TOTAL_BLOCKS/self.clients_num)  # 过小会导致时间过大，无法正常归一化。xlog(1+1/x)
            min_rb_lst.append(min_rb)
            max_rb_lst.append(max_rb)
        # 算max min T
        min_Ts = []
        max_Ts = []
        for i, client in enumerate(self.clients):  # 得到最大/小rate时的rb。（实验结果递增）
            client.set_rb_num(max_rb_lst[i])  # T min (rate max)
            min_Ts.append(client.get_cost()[0])
            client.set_rb_num(min_rb_lst[i])  # T max
            max_Ts.append(client.get_cost()[0])
        min_T, max_T = min(min_Ts), max(max_Ts)  # min也是近似。

        # 难以求解，近似。
        for client in self.clients:
            client.set_rb_num(int(TOTAL_BLOCKS / self.num_choose))  # (rate max)
        all_E = [client.get_cost()[1] for client in self.clients]
        max_E = max(all_E) * self.num_choose
        min_E = min(all_E) * self.num_choose

        print(f" * Clients: max_T:{max_T}, min_T:{min_T}, max_E:{max_E}, min_E:{min_E}")
        return max_T, min_T, max_E, min_E

    # 需要返回初始state和info。
    def reset(self, seed=SEED, options=None):
        # 获得quality状态。由于客户端模型会变化，需要重新重置：严禁用测试集训练数据。
        # 经过梯度下降算法，每次reset得到的值并不是一样的。所以是一个变化的状态
        self._reset_clients()
        self.qualities = self._get_data_qualities()
        self._reset_clients()

        # 重置：精确度、模型参数、当前轮数、参与次数
        self.latest_acc = 0
        self.current_round = 0
        self._set_model_parameters(self.original_model.state_dict())
        self.latest_global_model = self._get_model_parameters()
        self.parti_times = np.zeros(self.clients_num)

        # qualities = self._get_data_qualities()
        blank = np.zeros(self.clients_num)
        self.history_allocs = np.zeros((self.clients_num, self.window_size))
        self.all_allocations = blank
        self.all_energies = blank
        self.all_times = blank
        info = {  # must dict
            "current_round": self.current_round,
            "clients": self.clients,
            "parti_times": self.parti_times,
            "data_sizes": self.data_sizes,
            "data_qualities": self.data_qualities,
            # "data_sizes": self.data_sizes,
            # "data_qualities": self.qualities,
        }
        obs = {
            "data_qualities": zscore_normalize(self.qualities),
            "data_sizes": zscore_normalize(self.data_sizes),
            "gains": self.gains,
            "frequencies": self.frequencies,
            "powers": self.powers,
            "parti_times": self.parti_times / self.global_rounds,
            "allocations": self.history_allocs,
            # 最近一次分配的带宽
            # "all_allocations": blank,
            # "all_times": blank,
            # "all_energies": blank,
            # "allocations": np.ones(self.clients_num) / self.clients_num,
            # "times": [1] * self.clients_num,
            # "energys": [1] * self.clients_num,
            # "current_round": self.current_round,
        }
        # print(obs)
        # self.observation.update(obs)
        # obs = self.dict_to_vector(self.observation)
        # print(self.min_single_E,self.max_single_E)
        # exit()
        return obs, info

    """
    #0.各客户端完成数据集的初始化。服务端和客户端初始化一个分类模型。
    #1.**选取客户端，并为选择的客户端分配带宽**
    #2.被选中的客户端集合，使用本地数据集和收到的全局模型参数进行训练固定epoch，得到更新后的本地参数
    #3.上传梯度。**梯度有噪声**。
    #4.服务端全局聚合，更新参数。然后下发（也或许可以考虑噪声？），回到1。
    action: 2xK . [[indices][allocs]]    
    """

    def step(self, action):
        print("\n========")
        print(f"is fed train:{self.is_fed_train}")

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if self.task == 'acc':
            print(action)
            alc = np.full_like(action, 0.1, dtype=np.float32)
            action = np.stack([action, alc], axis=0)  # 2xK

        indices = np.round(action[0]).astype(np.int32).tolist()
        print("select action:", indices)
        allocs = np.floor(action[1] * self.total_bandwidth).astype(np.int32)
        print("alloc action:", allocs)
        if allocs.sum() > self.total_bandwidth:
            raise ValueError('bandwidth sum error')

        args = self.args

        # ==== 匈牙利匹配=========
        # if self.is_use_hungarian:
        #     cost_matrix = np.zeros((self.num_choose, self.num_choose))
        #     select_clients = [self.clients[i] for i in indices]
        #     for i, client in enumerate(select_clients):
        #         for j, rb_num in enumerate(allocs):
        #             client.set_rb_num(max(1, rb_num))  # 确保最小资源为1
        #             T, E = client.get_cost()
        #             time_rew = (T - self.min_T) / (self.max_T - self.min_T)
        #             energy_rew = (E - self.min_E) / (self.max_E - self.min_E)
        #             cost = -args.rew_b * time_rew - args.rew_c * energy_rew
        #             cost_matrix[i][j] = -cost  # 匈牙利算法是最小化cost

        #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #     best_alloc = np.zeros(self.num_choose, dtype=np.int32)
        #     for i, j in zip(row_ind, col_ind):
        #         best_alloc[i] = max(1, allocs[j])
        #     print("matches alloc:")
        #     print(best_alloc)
        #     allocs = best_alloc
        # ====================

        # === 联邦过程 ==
        # 1.已经选取了客户端和带宽(action)
        for idx in indices:
            self.parti_times[idx] += 1

        # 2. 本地训练，获得参数集合和各自的损失 | 接收到的参数集合是加噪声的
        # stats[i]：dict,  keys： [loss accuracy time energy]
        local_model_paras_set, stats = self.local_train(indices, allocs,
                                                        self.is_fed_train)  # 返回加噪后上传用的噪声，同时client的模型参数也类内更新
        # 获得当前的T和E:
        times = np.zeros(self.clients_num)
        energys = np.zeros(self.clients_num)
        all_clients_allocs = np.zeros(self.clients_num)
        select_times = [stat["time"] for stat in stats]
        select_energys = [stat["energy"] for stat in stats]
        times[indices] = select_times  # 记录所有客户端的时间。没被选中就是0.
        energys[indices] = select_energys  # 和能量
        all_clients_allocs[indices] = allocs  # 以及分配
        total_time = max(select_times)  # 该轮联邦过程时间
        total_energy = sum(select_energys)  # 和能耗
        # 3.使用加噪后的参数全局聚合
        averaged_paras = self.aggregate_parameters(local_model_paras_set)
        # 4.全局模型参数更新、分发全局模型参数
        self._set_model_parameters(averaged_paras)
        self._set_clients_parameters(averaged_paras)
        # 5.获得数据质量状态量。必须再更新回去，严禁在联邦训练过程中用测试集训练。
        # qualities = self._get_data_qualities()
        # self._set_clients_parameters(averaged_paras)

        self.latest_global_model = averaged_paras
        # 用获得的全局参数测试模型,[loss acc]
        stats_from_test_data = self.test_latest_model_on_testdata()
        global_loss, global_acc = (
            stats_from_test_data["loss"],
            stats_from_test_data["acc"],
        )

        self.current_round += 1

        # =======================
        # update clients
        for client in self.clients:
            client.update_state()

        # ======================
        # == 设计奖励 & 返回值 ==

        # t and e cost
        if args.norm_cost:
            time_rew = (total_time - self.min_T) / (self.max_T - self.min_T)  # min-max normalize
            energy_rew = (total_energy - self.min_E) / (self.max_E - self.min_E)
        else:
            time_rew = total_time
            energy_rew = total_energy

        # acc_rew
        if self.is_fed_train:
            if args.acc_delta:
                acc_rew = acc_func(global_acc, self.dataset_name) - acc_func(self.latest_acc, self.dataset_name)
            else:
                acc_rew = np.power(64, global_acc)
        else:
            gain = sum(self.data_sizes[i] / self.data_qualities[i] for i in indices)
            gain_all = sum(self.data_sizes[i] / self.data_qualities[i] for i in range(self.clients_num))
            acc_rew = gain / gain_all
        # acc_rew = global_acc

        # == parti penalty
        if indices == []:
            penalty = 1
        else:
            # pscore_max = ((1 + self.global_rounds)**args.penalty_coef - 1)  # * len(indices) *
            # pscore = sum((1 + self.parti_times[i])**args.penalty_coef - 1 for i in indices)
            # penalty = pscore / (pscore_max * self.num_choose)
            penalty = sum((1 + self.parti_times[i]/self.global_rounds)**args.penalty_coef - 1 for i in indices)

        if self.task == 'acc':
            reward = args.rew_a * acc_rew - args.rew_d * penalty
        elif self.task == 'hybrid':
            reward_cost = args.rew_b * time_rew + args.rew_c * energy_rew / self.num_choose
            reward = args.rew_a * acc_rew - reward_cost - args.rew_d * penalty

        else:
            raise ValueError("task not supported")

        print(
            f"reward: {reward} total_time: {total_time} total_energy: {total_energy} reward T&E: {args.rew_b*time_rew}, {args.rew_c * energy_rew / self.num_choose}\n\
              global acc: {global_acc} last global acc: {self.latest_acc} acc_rew: {args.rew_a * acc_rew} \n\
              penalty: {penalty} parti_times: {self.parti_times} \n\
              abcd:{args.rew_a},{args.rew_b},{args.rew_c},{args.rew_d}")
        # ======================

        self.latest_acc = global_acc
        # ==log
        if self.is_log == True:
            self.log(f"=====\n{self.current_round} / {self.global_rounds} round")
            self.log("action: " + str(action))
            self.log(f"total T AND E:{total_time}, {total_energy}")
            self.log(f"reward T AND E:{time_rew}, {energy_rew}")
            for idx, stat in enumerate(stats):
                loss, acc, t, e = (
                    stat["loss"],
                    stat["accuracy"],
                    stat["time"],
                    stat["energy"],
                )
                self.log(f"Client {indices[idx]} loss:{loss} accuracy:{acc} time:{t} energy:{e}")
            self.log(f"global loss:{global_loss} global accuracy:{global_acc}")
            self.log("\n")

        # step函数返回：observation, reward, done, done ,info
        # observation：[D,q,G,P,b_,T_,E_]
        # =========== 更新obs
        # self.all_allocations = (self.all_allocations + all_clients_allocs / self.total_bandwidth)
        # self.all_times = self.all_times + times
        # self.all_energies = self.all_energies + energys
        self.history_allocs[:, :-1] = self.history_allocs[:, 1:]  # numpy has no pop()
        self.history_allocs[:, -1] = 0
        self.history_allocs[indices, -1] = action[1]
        obs = {
            "data_qualities": zscore_normalize(self.data_qualities),
            "data_sizes": zscore_normalize(self.data_sizes),
            "gains": zscore_normalize([client.attr_dict["gain"] for client in self.clients]),
            "frequencies": self.frequencies,
            "powers": self.powers,
            "parti_times": self.parti_times / self.global_rounds,
            "allocations": self.history_allocs,
        }
        # print(obs)
        # self.observation.update(obs)
        # observasion = self.dict_to_vector(self.observation)

        info = {
            "clients": self.clients,
            "current_round": self.current_round,
            "data_sizes": self.data_sizes,
            "data_qualities": self.data_qualities,
            "global_accuracy": global_acc,
            "total_time": total_time,
            "total_energy": total_energy,
            "parti_times": self.parti_times,
            "reward": reward,
        }
        # print(obs)
        # print(observasion)
        done = False
        if self.current_round >= self.global_rounds:
            done = True
        if self.is_fed_train and global_acc >= self.max_acc[self.dataset_name]:
            done = True
        return obs, reward, done, done, info  # 新gym返回五个值。

    def log(self, str):
        with open(self.log_file, "a") as f:
            f.write(str + "\n")
        print(str)

    def close(self):
        print("federated progress done...")

    def local_train(self, indices, allocs, is_need_train=True):
        local_model_paras_set = []
        stats = []
        for client_index, rb_num in zip(indices, allocs):
            client = self.clients[client_index]
            client.set_model_parameters(self.latest_global_model)
            if rb_num > 0:  # 只有当分配的资源块大于 0 时才训练
                client.set_rb_num(rb_num)
            else:
                client.set_rb_num(1)
            local_model_paras, stat = client.local_train(is_need_train=is_need_train, local_rounds=self.local_rounds)
            # 参数加噪
            if self.args.add_noise:
                noise_std = self.base_noise / np.sqrt(rb_num * B + self.noise_epsilon)
                noisy_params = self.add_noise(local_model_paras, noise_std)
                local_model_paras_set.append((len(client.local_dataset), noisy_params))
            else:
                local_model_paras_set.append((len(client.local_dataset), local_model_paras))
            stats.append(stat)
        return local_model_paras_set, stats

    def aggregate_parameters(self, local_model_paras_set):
        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = sum(num_samples for num_samples, _ in local_model_paras_set)
        for var in averaged_paras:
            averaged_paras[var] = (sum(num_samples * local_model_paras[var]
                                       for num_samples, local_model_paras in local_model_paras_set) / train_data_num)
        return averaged_paras

    def test_latest_model_on_testdata(self):
        self._set_model_parameters(self.latest_global_model)
        testData = self.dataset.test_data
        test_loader = DataLoader(testData, batch_size=10)
        test_loss, test_acc, test_total = 0.0, 0.0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (pred.argmax(1) == y).sum().item()
                test_total += y.size(0)

        stats = {"loss": test_loss / test_total, "acc": test_acc / test_total}
        return stats

    def add_noise(self, params, noise_std):
        """为模型参数添加高斯噪声"""
        noisy_params = {}
        for k, v in params.items():
            v_float = v.float()  # 转换为 float 类型
            noise = torch.randn_like(v_float) * noise_std
            noisy_params[k] = v + noise
        return noisy_params

    def dict_to_vector(self, observation):  # tianshou not support dict observation
        vector = []
        for key, value in observation.items():
            if isinstance(value, (int, float)):
                # 单个值直接加入
                vector.append(value)
            elif isinstance(value, (list, np.ndarray)):
                value = np.array(value, dtype=np.float32)
                vector.extend(value)
        return np.array(vector, dtype=np.float32)


if __name__ == "__main__":
    # example usage:
    import random

    args = get_args()
    np.random.seed(args.seed)
    dataset = FedDataset(args)
    model = choose_model(args)

    attr_dicts = init_attr_dicts(args.num_clients)

    # == 初始化客户端数据集
    clients = init_clients(args.num_clients, model, dataset, attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
    data_distribution = dataset.get_data_distribution()
    print("Data distribution:")
    for client_id, dist in data_distribution.items():
        print(f"Client {client_id}: {dist},samples num:{sum(dist)}")

    # ==

    env = FederatedEnv(args, dataset, clients, model=model)
    state, info = env.reset()
    print("env reset!init state:", state)

    print("====\nstart training")
    global_acc = []
    for i in range(2):
        env.reset()
        print("global")
        for i in range(10):
            print("****round:", i)

            selects = random.sample(range(100), 10)
            selects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            # allocs = np.array([10, 15, 25, 30, 20])
            allocs = np.array([0.1]*10)
            action = np.vstack([selects, allocs])

            next_state, reward, done, done, info = env.step(action)
            # print(info['data_qualities'])
            # global_acc.append(info["global_accuracy"])
            # print("\nnext_state:", next_state)
            if done:
                break

    env.close()
    print(global_acc)

    # [0.7148, 0.7552, 0.91, 0.937, 0.9667, 0.9638, 0.9744, 0.974, 0.9776, 0.9788]
    # [0.5821, 0.6757, 0.6697, 0.9011, 0.9496, 0.9646, 0.9684, 0.9705, 0.9732, 0.9733]
