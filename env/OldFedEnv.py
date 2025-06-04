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


def minmax_normalize(x: np.ndarray, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)


def criterion(pred, y):
    return F.cross_entropy(pred, y)


class FederatedEnv(gym.Env):

    def __init__(self, args, dataset: FedDataset, clients: list[Client], model):
        super().__init__()

        # == log fed progress.irrelevant with RL. ==
        from datetime import datetime
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.log_file = f"./logs/log_{timestamp}.txt"
        self.is_log = args.log_fed
        # ==
        self.original_model = copy.deepcopy(model)

        self.model = model.cuda()  # 全局模型

        self.args = args
        self.dataset = dataset
        # 用于决定训练RL时是否需要真正训练。
        self.is_fed_train = False
        if args.evaluate:
            self.is_fed_train = True
        # 初始化客户端以及对应的数据集。
        self.clients = clients

        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.local_rounds = args.local_rounds  # 训练轮数
        self.global_rounds = args.global_rounds
        self.latest_acc = 0

        self.clients_num = len(self.clients)
        self.num_choose = args.num_choose
        self.parti_times = [0] * self.clients_num  # 参与次数.

        self.latest_global_model = self.get_model_parameters()
        self.current_round = 0
        self.data_sizes = [len(client.local_dataset) for client in self.clients]
        self.gains = [client.attr_dict["gain"] for client in self.clients]
        self.base_noise = args.base_noise
        self.noise_epsilon = 1e-5

        # 归一化用 . min T = min single T
        self.max_T, self.min_T, self.max_E, self.min_E, self.min_single_E, self.max_single_E = self._get_boundary_cost()
        self.data_qualities = self._get_data_quality()
        self.sum_qualities = sum(self.data_qualities)
        self.total_bandwidth = TOTAL_BLOCKS

        self.all_allocations = np.zeros(self.clients_num)
        self.all_times = np.zeros(self.clients_num)
        self.all_energies = np.zeros(self.clients_num)
        # STATE ACTION 的SPACE
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.clients_num,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "data_qualities": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.clients_num,),
                    dtype=np.float32,
                ),
                "data_sizes": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.clients_num,),
                    dtype=np.float32,
                ),
                "parti_times": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.clients_num,),
                    dtype=np.float32,
                ),
                "gains": spaces.Box(
                    low=-100, high=100, shape=(self.clients_num,), dtype=np.float32
                ),
                "alloc": spaces.Box(
                    low=0, high=1.0, shape=(self.clients_num,), dtype=np.float32
                ),
                # 'times':
                # spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32),
                # 'energys':
                # spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32)
            }
        )
        self.observation = {}

    # only use in init:  在刚开始时获取数据质量。需要先计算本地损失值。
    def _get_data_quality(self):
        # 记录每个客户端的初始 loss 和训练时间
        losses = []
        for client in self.clients:
            local_model_paras, stat = client.local_train()
            losses.append(stat["loss"])
        qualities = [
            # (1 + np.log(size)) / (1 + np.sqrt(loss))
            1 / loss
            for size, loss in zip(self.data_sizes, losses)
        ]
        print("data qualities:")
        print("init losses:", losses)
        print("data_sizes:", self.data_sizes)
        print("qualities:", qualities)
        # # 节约测试时间用的。
        # qualities = [5.593671702102716, 5.827380072319978, 5.166727750007433, 4.468516490857968, 5.683148413011656,
        #              6.4199849554002935, 5.848096529012861, 5.723686200111781, 5.174420441064506, 5.62497748431034]

        return qualities

    def get_model_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_parameters(self, model_parameters_dict):
        self.model.load_state_dict(model_parameters_dict)

    # get min-max T&E. only used in init
    def _get_boundary_cost(self):
        min_rb_lst = []
        max_rb_lst = []
        for client in self.clients:  # 得到最大/小rate时的rb。（实验结果递增）
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
        # print(min_rb_lst, max_rb_lst) #1.. 100..
        # 算max min T
        max_T, min_T, max_E, min_E = 0, 0, 0, 0
        for client in self.clients:  # 得到最大/小rate时的rb。（实验结果递增）
            client.set_rb_num(max_rb)  # T min (rate max)
        all_T, all_E = zip(*[client.get_cost() for client in self.clients])
        min_T, min_E, min_single_E = min(all_T), min(all_E)*10, min(all_E)
        # print(all_T,min_T,all_E,min_E)

        # ===== 实际实验中，发现带宽分配不会出现为min_rb的极端情况，因此需要做近似以保证实验。
        for client in self.clients:
            client.set_rb_num(int(TOTAL_BLOCKS/self.clients_num))  # T min (rate max)
        all_T, all_E = zip(*[client.get_cost() for client in self.clients])
        max_T, max_E, max_single_E = max(all_T), sum(all_E), max(all_E)

        # for client in self.clients:
        #     client.set_rb_num(min_rb)  # T max
        # _, all_E = zip(*[client.get_cost() for client in self.clients])
        # max_E = max(all_E)  # min_E最小，但max_E是近似的。即便如此，rew的惩罚项也足够高。
        # print(all_T,max_T,all_E,max_E)

        print(f" * Clients: max_T:{max_T}, min_T:{min_T}, max_E:{max_E}, min_E:{min_E}")
        return max_T, min_T, max_E, min_E, min_single_E, max_single_E

    # 需要返回初始state和info。
    def reset(self, seed=SEED, options=None):
        # 重置各自客户端模型 以及 全局模型
        for client in self.clients:
            client.set_model_parameters(self.original_model.state_dict())
            client.set_optim()

        # 重置：精确度、模型参数、当前轮数、参与次数
        self.latest_acc = 0
        self.current_round = 0
        self.set_model_parameters(self.original_model.state_dict())
        self.latest_global_model = self.get_model_parameters()
        self.parti_times = [0] * self.clients_num
        blank = np.zeros(self.clients_num)
        self.all_allocations = blank
        self.all_energies = blank
        self.all_times = blank
        info = {"current_round": self.current_round}  # must dict and same as step!~
        obs = {
            "data_qualites": self.data_qualities,
            "data_sizes": self.data_sizes,
            "gains": self.gains,
            "parti_times": self.parti_times,
            # 最近一次分配的带宽
            "all_allocations": blank,
            "all_times": blank,
            "all_energies": blank,
            # "allocations": np.ones(self.clients_num) / self.clients_num,
            # "times": [1] * self.clients_num,
            # "energys": [1] * self.clients_num,
            # "current_round": self.current_round,
        }
        self.observation.update(obs)
        obs = self.dict_to_vector(self.observation)
        # print(self.min_single_E,self.max_single_E)
        # exit()
        return obs, info

    """
    #0.各客户端完成数据集的初始化。服务端和客户端初始化一个分类模型。
    #1.**选取客户端，并为选择的客户端分配带宽**
    #2.被选中的客户端集合，使用本地数据集和收到的全局模型参数进行训练固定epoch，得到更新后的本地参数
    #3.上传梯度。**梯度有噪声**。
    #4.服务端全局聚合，更新参数。然后下发（也或许可以考虑噪声？），回到2。
    action: 2xK . [[indices][allocs]]    
    """

    def step(self, action):
        print("\n========")
        print(f"is fed train:{self.is_fed_train}")

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        indices = np.round(action[0]).astype(np.int32).tolist()
        print("select action:", indices)
        allocs = np.floor(action[1] * self.total_bandwidth).astype(np.int32)
        print("alloc action:", allocs)
        if allocs.sum() > self.total_bandwidth:
            raise ValueError('bandwidth sum error')

        args = self.args

        # ==== 匈牙利匹配=========
        cost_matrix = np.zeros((self.num_choose, self.num_choose))
        select_clients = [self.clients[i] for i in indices]
        for i, client in enumerate(select_clients):
            for j, rb_num in enumerate(allocs):
                client.set_rb_num(max(1, rb_num))  # 确保最小资源为1
                T, E = client.get_cost()
                time_rew = (T - self.min_T) / (self.max_T - self.min_T)
                energy_rew = (E - self.min_E) / (self.max_E - self.min_E)
                cost = - args.rew_b * time_rew - args.rew_c * energy_rew
                cost_matrix[i][j] = -cost  # 匈牙利算法是最小化cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_alloc = np.zeros(self.num_choose, dtype=np.int32)
        for i, j in zip(row_ind, col_ind):
            best_alloc[i] = max(1, allocs[j])
        print("matches alloc:")
        print(best_alloc)
        allocs = best_alloc
        # ====================

        # === 联邦过程 ==
        # 1.已经选取了客户端和带宽(action)
        print("select action", indices)
        for idx in indices:
            self.parti_times[idx] += 1

        # 2. 本地训练，获得参数集合和各自的损失
        # 3. 获得的参数集合加噪声
        # stats[i]：dict,  keys： [loss accuracy time energy]
        local_model_paras_set, stats = self.local_train(indices, allocs, self.is_fed_train)
        # 获得当前的T和E:
        times = np.zeros(self.clients_num)
        energys = np.zeros(self.clients_num)
        all_clients_allocs = np.zeros(self.clients_num)
        select_times = [stat["time"] for stat in stats]
        select_energys = [stat["energy"] for stat in stats]
        times[indices] = select_times
        energys[indices] = select_energys
        all_clients_allocs[indices] = allocs
        total_time = max(select_times)
        total_energy = sum(select_energys)
        # 4.全局聚合，更新全局模型参数
        averaged_paras = self.aggregate_parameters(local_model_paras_set)
        self.set_model_parameters(averaged_paras)
        self.latest_global_model = averaged_paras
        # 用获得的全局参数测试模型,[loss acc]
        stats_from_test_data = self.test_latest_model_on_testdata()
        global_loss, global_acc = (
            stats_from_test_data["loss"],
            stats_from_test_data["acc"],
        )

        self.current_round += 1
        # update clients
        for client in self.clients:
            client.update_state()

        # ======================
        # == 设计奖励 & 返回值 ==
        time_rew = (total_time - self.min_T) / (
            self.max_T - self.min_T
        )  # min-max normalize
        energy_rew = (total_energy - self.min_E) / (self.max_E - self.min_E)
        del_acc = global_acc - self.latest_acc
        # original_del = del_acc
        # if self.current_round == 1:  # 0.50~70->10
        #     del_acc = del_acc * 20
        # elif self.current_round <= 3:
        #     del_acc = del_acc * 100
        # else:
        #     del_acc = del_acc * 500
        data_qualities_gain = (
            sum([self.data_qualities[i] for i in indices]) / self.sum_qualities
        )
        self.latest_acc = global_acc
        # == parti penalty
        if indices == []:
            penalty = 1
        else:
            pscore_max = ((1 + self.global_rounds) ** args.penalty_coef - 1)  # * len(indices) *
            pscore = sum((1 + self.parti_times[i]) ** args.penalty_coef - 1 for i in indices)
            penalty = pscore / pscore_max

        reward = (  # a*gain - (1-a)[bt+(1-b)e] - c*penalty
            args.rew_a * data_qualities_gain - args.rew_b * time_rew - args.rew_c * energy_rew - args.rew_d * penalty) * 100
        print(
            f"reward: {reward} del_acc: {del_acc} total_time: {total_time} total_energy: {total_energy} reward T&E: {time_rew}, {energy_rew}\n\
              global acc: {global_acc} rew_T&rew_E: {-time_rew},{-energy_rew} rew_quality: {data_qualities_gain} penalty: {penalty} parti_times: {self.parti_times} \n\
              abcd:{args.rew_a},{args.rew_b},{args.rew_c},{args.rew_d}")
        # ======================

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
                self.log(
                    f"Client {indices[idx]} loss:{loss} accuracy:{acc} time:{t} energy:{e}"
                )
            self.log(f"global loss:{global_loss} global accuracy:{global_acc}")
            self.log("\n")

        # step函数返回：observation, reward, done, done ,info
        # observation：[L,D,N,h,T,E;t] 即：
        # 数据标签程度数据集大小 历史参与次数 信道增益; 各时间，能耗；当前时间步 等等...
        # 根据oort：如果loss越大，那么就越值得被选用。
        # not been converted to tensor
        # losses = self.restore_array(  # /0 ERROR!!
        #     self.clients_num, indices, [stat["loss"] for stat in stats], is_norm=True
        # )
        # times = self.restore_array(
        #     self.clients_num, indices, [stat["time"] for stat in stats], is_norm=True
        # )
        # energys = self.restore_array(
        #     self.clients_num, indices, [stat["energy"] for stat in stats], is_norm=True
        # )
        # only update dynamic.

        # =========== 更新obs
        self.all_allocations = (self.all_allocations + all_clients_allocs / self.total_bandwidth)
        self.all_times = self.all_times + times
        self.all_energies = self.all_energies + energys
        obs = {
            "data_qualites": self.data_qualities,
            "data_sizes": self.data_sizes,
            "gains": [client.attr_dict['gain'] for client in self.clients],
            # "allocations": self.observation["allocations"],
            "parti_times": self.parti_times,
            # "times": self.observation['times'],
            # "energys": self.observation["energys"]
            "all_allocations": self.all_allocations / self.current_round,
            "all_times": minmax_normalize(self.all_times, 0),
            "all_energies": minmax_normalize(self.all_energies, 0)
        }
        # 只更新被选中的客户端状态
        # for i, client_idx in enumerate(indices):
        #     obs["allocations"][client_idx] = allocs[i]  # 更新带宽分配比例
        #     obs["times"][client_idx] = (stats[i]["time"] - self.min_T) / (self.max_T - self.min_T)
        #     obs["energys"][client_idx] = (
        #         stats[i]["energy"] - self.min_single_E) / (self.max_single_E - self.min_single_E)

        self.observation.update(obs)

        observasion = self.dict_to_vector(obs)

        # self.all_time += total_time
        # self.all_energy += total_energy
        info = {
            "current_round": self.current_round,
            "global_loss": global_loss,
            "global_accuracy": global_acc,
            "total_time": total_time,
            "total_energy": total_energy,
            # "env_all_time": self.all_time,
            # "env_all_energy": self.all_energy,
        }
        print(obs)
        # print(observasion)
        done = self.current_round >= self.global_rounds
        return observasion, reward, done, done, info  # 新gym返回五个值。

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
            local_model_paras, stat = client.local_train(is_need_train=is_need_train)
            # 参数加噪
            noise_std = self.base_noise / np.sqrt(rb_num * B + self.noise_epsilon)
            noisy_params = self.add_noise(local_model_paras, noise_std)
            local_model_paras_set.append((len(client.local_dataset), noisy_params))
            stats.append(stat)
        return local_model_paras_set, stats

    def aggregate_parameters(self, local_model_paras_set):
        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = sum(num_samples for num_samples, _ in local_model_paras_set)
        for var in averaged_paras:
            averaged_paras[var] = (
                sum(
                    num_samples * local_model_paras[var]
                    for num_samples, local_model_paras in local_model_paras_set
                )
                / train_data_num
            )
        return averaged_paras

    def test_latest_model_on_testdata(self):
        self.set_model_parameters(self.latest_global_model)
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

    # 恢复成原本大小数组（即K->N dim）  同时进行归一化
    def restore_array(self, N, indices, values, is_norm=True):
        arr = [values[indices.index(i)] if i in indices else 0 for i in range(N)]
        if is_norm:
            min_val = min(arr)  # 最小值
            max_val = max(arr)  # 最大值
            normalized_arr = [(x - min_val) / (max_val - min_val) for x in arr]
            return normalized_arr
        return arr

    def dict_to_vector(self, observation):  # tianshou not support dict observation
        vector = []
        for key, value in observation.items():
            if isinstance(value, (int, float)):
                # 单个值直接加入
                vector.append(value)
            elif isinstance(value, (list, np.ndarray)):
                # 对列表或数组进行标准化。对于某些特殊值特殊处理
                if key == "parti_times":
                    value = np.array(value, dtype=np.float32)
                    vector.extend(value / self.global_rounds)
                elif key.startswith("all_"):
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


# 必须包装成tianshou可以识别的向量环境。


def make_env(args, dataset, clients, model):
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
    env = FederatedEnv(args, dataset, clients, model)
    train_envs = DummyVectorEnv([lambda: env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: env for _ in range(args.test_num)])

    return env, train_envs, test_envs


if __name__ == "__main__":
    # example usage:
    import random
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

    env = FederatedEnv(args, dataset, clients, model=model)
    state, info = env.reset()
    print("env reset!init state:", state)

    print("====\nstart training")
    global_acc = []
    for i in range(1):
        env.reset()
        print("global")
        for i in range(10):
            print("****round:", i)

            selects = random.sample(range(10), 5)
            allocs = np.random.rand(5)
            allocs = allocs / allocs.sum()
            action = np.vstack([selects, allocs])

            next_state, reward, done, done, info = env.step(action)
            print(info["global_accuracy"])
            print(info["global_loss"])
            global_acc.append(info["global_accuracy"])
            # print("\nnext_state:", next_state)
            if done:
                break

    env.close()
    print(global_acc)
    # [0.5821, 0.6757, 0.6697, 0.9011, 0.9496, 0.9646, 0.9684, 0.9705, 0.9732, 0.9733]
