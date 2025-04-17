import copy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from tqdm import tqdm
from dataset import FedDataset
from config import *
from models import resnet18

criterion = F.cross_entropy
mse_loss = nn.MSELoss()


class Attribute():

    def __init__(self, cpu_frequency, transmit_power, gain, distance):
        self.cpu_frequency = cpu_frequency
        self.transmit_power = transmit_power
        self.gain = gain
        self.distance = distance

    def get_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "transmit_power": self.transmit_power[id],
            "distance": self.distance[id],
            "gain": self.gain[id]
        }


class Client():

    def __init__(self, id, dataset, model, attr_dict: dict, args):
        self.id = id
        self.local_dataset = dataset
        self.model = model
        self.attr_dict = attr_dict
        self.args = args
        self.rb_num = 0
        self.optimizer = None
        self.set_optim()

    def set_optim(self):
        args = self.args
        if args.fed_optim == 'adam':
            client_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.fed_lr)
        elif args.fed_optim == 'sgd':
            client_optimizer = torch.optim.SGD(self.model.parameters(), lr=args.fed_lr)
        else:
            raise ValueError("Invalid federated learning optimizer")
        self.optimizer = client_optimizer

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def calculate_model_size(self):
        total_bytes = 0
        state_dict = self.get_model_parameters()
        for param in state_dict.values():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    def set_model_parameters(self, model_parameters_dict):
        self.model.load_state_dict(model_parameters_dict)

    def local_train(self, is_need_train=True):
        localTrainDataLoader = DataLoader(self.local_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.model.cuda().train()
        train_loss = train_acc = train_total = 0
        mean_loss = acc = 0
        if self.args.log_client:
            print(f"client {self.id} training")
        # 如果RL奖励函数与损失、精度无关；且状态不包括loss，则不需要训练，只需要返回t和e
        # 但是测试的时候必须训练以获取精度。
        if is_need_train:
            for epoch in tqdm(range(self.args.local_rounds), desc="Epoch", disable=not self.args.log_client):
                for X, y in localTrainDataLoader:
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += y.size(0)
            mean_loss = train_loss / train_total
            acc =  train_acc / train_total
            
        local_model_paras = self.get_model_parameters()
        time, energy = self.get_cost()
        if self.args.log_client:
            print(f"loss: {train_loss / train_total}  time: {time}  energy: {energy}")
        return local_model_paras, {
            "loss": mean_loss,
            "accuracy": acc,
            "time": time,
            "energy": energy
        }

    # === 以下是成本计算相关
    def set_rb_num(self, rb_num):
        self.rb_num = rb_num

    def get_transmission_rate(self):
        p = self.attr_dict['transmit_power']
        h = self.attr_dict['gain']
        d = self.attr_dict['distance']
        alpha = -2
        N0 = 10**(-16)
        n = self.rb_num
        rate = (n * B) * np.log2(1 + (p * h * (d**-alpha)) / (N0 * n * B + 1e-6)) #avoid / 0

        return rate

    def get_communication_time(self):
        R_k = self.get_transmission_rate()
        Z_k = self.calculate_model_size() * 8  # bytes->bits
        return Z_k / R_k

    def get_computation_time(self):
        E = self.args.local_rounds
        D = len(self.local_dataset)
        C = self.args.cycles_per_sample
        f = self.attr_dict['cpu_frequency']
        return (E * C * D) / f

    def get_computation_energy(self):
        C = self.args.cycles_per_sample
        E = self.args.local_rounds
        D = len(self.local_dataset)
        f = self.attr_dict['cpu_frequency']
        k = 1e-28
        return k * C * E * D * (f**2)

    def get_communication_energy(self):
        T_com = self.get_communication_time()
        p = self.attr_dict['transmit_power']
        return T_com * p

    def get_cost(self):
        cmp_t, com_t = self.get_computation_time(), self.get_communication_time()
        cmp_e, com_e = self.get_computation_energy(), self.get_communication_energy()
        if self.args.log_client:
            print(f"client {self.id} cost:")
            print(f"cmp T & E : {cmp_t} & {cmp_e}")
            print(f"com T & E : {com_t} & {com_e}")
        T = cmp_t + com_t
        E = cmp_e + com_e
        return T, E
    
    def update_state(self):
        self.update_gain()
        # more to add .
        
    def update_gain(self):
        self.attr_dict['gain'] = np.random.normal(GAIN_MEAN, GAIN_STD)


# 初始化客户端,根据一个同长度的属性列表
def init_clients(clients_num, model, dataset, attr_dicts, args):
    assert len(attr_dicts) == clients_num, "attr_dicts length must equal to clients_num"
    clients = []
    for i in range(clients_num):
        # 为每个客户端创建独立的模型和优化器副本
        client_model = copy.deepcopy(model)
        clients.append(Client(i, dataset, client_model, attr_dicts[i], args))
    return clients


# 初始化属性列表 根据高斯分布
def init_attr_dicts(client_num):
    cpu_frequency = np.random.normal(CPU_FREQUENCY_MEAN, CPU_FREQUENCY_STD, client_num)
    transmit_power = np.random.normal(TRANSMIT_POWER_MEAN, TRANSMIT_POWER_STD, client_num)
    gain = np.random.normal(GAIN_MEAN, GAIN_STD, client_num)
    distance = np.random.normal(DISTANCE_MEAN, DISTANCE_STD, client_num)

    cpu_frequency = np.clip(cpu_frequency, 1.5e9, 2.5e9)
    transmit_power = np.clip(transmit_power, 0.4, 0.6)
    gain = np.clip(gain, 4e-8, 6e-8)

    attr = Attribute(cpu_frequency, transmit_power, gain, distance)
    return [attr.get_attr(i) for i in range(client_num)]


if __name__ == '__main__':
    args = get_args()
    client_num = args.num_clients
    dataset = FedDataset(args)
    attr_dicts = init_attr_dicts(client_num)
    model = resnet18()
    clients = init_clients(client_num, model, dataset, attr_dicts, args)
    for client in clients:
        local_model_params, metrics = client.local_train()

    print("Training completed for all clients.")
