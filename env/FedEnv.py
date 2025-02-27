import torch
import time
from tianshou.env import DummyVectorEnv
from client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import argparse  # 添加 argparse 模块
import gym
from gym import spaces
from dataset import FedDataset
import torch.nn.functional as F
from model import *
from client import *


def criterion(pred, y):
    return F.cross_entropy(pred, y)


class FederatedEnv(gym.Env):

    def __init__(self, args, dataset: FedDataset, clients: list[Client], model=None, name=''):
        super().__init__()

        # == log
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
        # 初始化客户端以及对应的数据集。
        self.clients = clients

        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.local_rounds = args.local_rounds  # 训练轮数
        self.global_rounds = args.global_rounds
        self.latest_acc = 0

        self.clients_num = len(self.clients)
        self.parti_times = [0] * self.clients_num  # 参与次数
        self.name = '_'.join([name, f'wn{int(args.per_round_c_fraction * self.clients_num)}', f'tn{len(self.clients)}'])
        self.latest_global_model = self.get_model_parameters()
        self.current_round = 0
        self.data_sizes = [len(client.local_dataset) for client in self.clients]
        self.gains = [client.attr_dict['gain'] for client in self.clients]

        # STATE ACTION 的SPACE
        self.action_space = spaces.Box(low=0, high=1, shape=(self.clients_num, ), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'round':
            spaces.Discrete(1),
            'losses':
            spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32),
            'data_sizes':
            spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32),
            'parti_times':
            spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32),
            'gains':
            spaces.Box(low=-100, high=100, shape=(self.clients_num, ), dtype=np.float32),
            'times':
            spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32),
            'energys':
            spaces.Box(low=0, high=float('inf'), shape=(self.clients_num, ), dtype=np.float32)
        })

    def get_model_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_parameters(self, model_parameters_dict):
        self.model.load_state_dict(model_parameters_dict)

    # 需要返回初始state和info。
    def reset(self, seed=42, options=None):
        # 重置模型
        for client in self.clients:
            client.set_model_parameters(self.original_model.state_dict())
            client.set_optim()

        self.latest_acc = 0
        self.set_model_parameters(self.original_model.state_dict())
        self.current_round = 0
        self.latest_global_model = self.get_model_parameters()
        self.parti_times = [0] * self.clients_num
        blank = [0] * self.clients_num
        info = {'current_round': self.current_round}  # must dict and same as step!~
        observasion = {
            'losses': blank,
            'data_sizes': self.data_sizes,
            'parti_times': self.parti_times,
            'gains': self.gains,
            'times': blank,
            'energys': blank,
            'current_round': self.current_round
        }
        observasion = self.dict_to_vector(observasion)
        return observasion, info

    '''
    #0.各客户端完成数据集的初始化。服务端和客户端初始化一个分类模型。

    #1.**选取客户端，并为选择的客户端分配带宽**

    #2.被选中的客户端集合，使用本地数据集和收到的全局模型参数进行训练固定epoch，得到更新后的本地参数

    #3.上传梯度。**梯度有噪声**。

    #4.服务端全局聚合，更新参数。然后下发（也或许可以考虑噪声？），回到2。
    '''

    def step(self, action):
        args = self.args
        # 1.已经选取了客户端和带宽(action)
        action = np.array(action)  # 不然np.where返回的是数
        print("\n==action==\n", action)
        self.parti_times += np.where(action != 0, 1, 0)
        indices = [i for i, value in enumerate(action) if value != 0]

        # 2. 本地训练，获得参数集合和各自的损失
        # 3. 获得的参数集合加噪声（没实现）
        # stats[i]：dict,  keys： [loss accuracy time energy]
        local_model_paras_set, stats = self.local_train(action)

        # 获得当前的T和E:
        total_time = max(stat["time"] for stat in stats)
        total_energy = sum(stat["energy"] for stat in stats)

        # 4.全局聚合，更新全局模型参数
        averaged_paras = self.aggregate_parameters(local_model_paras_set)
        self.set_model_parameters(averaged_paras)
        self.latest_global_model = averaged_paras

        # 用获得的全局参数测试模型,[loss acc]
        stats_from_test_data = self.test_latest_model_on_testdata(self.current_round)
        global_loss, global_acc = stats_from_test_data['loss'], stats_from_test_data['acc']

        self.current_round += 1

        # == 设计奖励 & 返回值 ==
        del_acc = stats_from_test_data['acc'] - self.latest_acc
        original_del = del_acc
        if self.current_round == 1:  # 0.50~70->10
            del_acc = del_acc * 20
        elif self.current_round <= 3:
            del_acc = del_acc * 100
        else:
            del_acc = del_acc * 500
        self.latest_acc = stats_from_test_data['acc']
        reward = (args.rew_alpha * del_acc * 1.5 - args.rew_beta * total_time - args.rew_gamma * total_energy * 2) * 10
        print(f"reward: {reward} del_acc: {original_del} total_time: {total_time} total_energy: {total_energy}")

        # log
        if self.is_log == True:
            self.log(f"=====\n{self.current_round} / {self.global_rounds} round")
            self.log("action: " + str(action))
            self.log(f"total T AND E:{total_time}, {total_energy}")
            for idx, stat in enumerate(stats):
                loss, acc, t, e = stat['loss'], stat['accuracy'], stat['time'], stat['energy']
                self.log(f"Client {indices[idx]} loss:{loss} accuracy:{acc} time:{t} energy:{e}")
            self.log(f'global loss:{global_loss} global accuracy:{global_acc}')
            self.log('\n')

        # step函数返回：observation, reward, done, done ,info
        # observation：[L,D,N,h,T,E;t] 即：
        # 数据标签程度数据集大小 历史参与次数 信道增益; 各时间，能耗；当前时间步 等等...
        # 根据oort：如果loss越大，那么就越值得被选用。
        # not been converted to tensor
        losses = self.restore_array(self.clients_num, indices, [stat['loss'] for stat in stats])
        times = self.restore_array(self.clients_num, indices, [stat['time'] for stat in stats])
        energys = self.restore_array(self.clients_num, indices, [stat['energy'] for stat in stats])
        observasion = {
            'losses': losses,
            'data_sizes': self.data_sizes,
            'parti_times': self.parti_times,
            'gains': self.gains,
            'times': times,
            'energys': energys,
            'current_round': self.current_round
        }
        observasion = self.dict_to_vector(observasion)
        done = self.current_round >= self.global_rounds
        info = {
            'current_round': self.current_round,
            'global_loss': global_loss,
            'global_accuracy': global_acc,
            'total_time': total_time,
            'total_energy': total_energy,
        }
        return observasion, reward, done, done, info  # 新gym返回五个值。

    def log(self, str):
        with open(self.log_file, 'a') as f:
            f.write(str + "\n")
        print(str)

    def close(self):
        print("federated progress done...")

    def local_train(self, action):
        local_model_paras_set = []
        stats = []
        for client, rb_num in zip(self.clients, action):
            if rb_num > 0:  # 分配了资源块就训练
                client.set_model_parameters(self.latest_global_model)
                client.set_rb_num(rb_num)
                local_model_paras, stat = client.local_train()
                local_model_paras_set.append((len(client.local_dataset), local_model_paras))  # 要加样本数
                stats.append(stat)
        return local_model_paras_set, stats

    def aggregate_parameters(self, local_model_paras_set):
        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = sum(num_samples for num_samples, _ in local_model_paras_set)
        for var in averaged_paras:
            averaged_paras[var] = sum(num_samples * local_model_paras[var]
                                      for num_samples, local_model_paras in local_model_paras_set) / train_data_num
        return averaged_paras

    def test_latest_model_on_testdata(self, round_i):
        self.set_model_parameters(self.latest_global_model)
        testData = self.dataset.test_data
        test_loader = DataLoader(testData, batch_size=10)
        test_loss, test_acc, test_total = 0., 0., 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                test_loss += loss.item() * y.size(0)
                test_acc += (pred.argmax(1) == y).sum().item()
                test_total += y.size(0)

        stats = {'loss': test_loss / test_total, 'acc': test_acc / test_total}
        return stats

    def select_clients(self, action):
        return [self.clients[action]]

    # 恢复成原本大小数组（即K->N dim）
    # 同时进行归一化
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
                vector.append(value)
            elif isinstance(value, (list, np.ndarray)):
                vector.extend(value)
        return np.array(vector, dtype=np.float32)


# 必须包装成tianshou可以识别的向量环境。
def make_env(args, dataset, clients, model):
    """Wrapper function for env.
    :return: a tuple of (single env, training envs, test envs).
    """
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


if __name__ == '__main__':
    # example usage:
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
    for i in range(2):
        env.reset()
        print("global")
        while True:
            print("****round:", i)
            action = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
            next_state, reward, done, done, info = env.step(action)
            print(info['global_accuracy'])
            print(info['global_loss'])
            # print("\nnext_state:", next_state)
            if done:
                break

    env.close()

    #
