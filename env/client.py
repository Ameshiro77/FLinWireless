from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from tqdm import tqdm
from dataset import FedDataset
from config import *
from model import resnet18

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

    def __init__(self, id, dataset, optim, model, attr_dict: dict, args):
        self.id = id
        self.local_dataset = dataset
        self.model = model
        self.optimizer = optim
        self.attr_dict = attr_dict
        self.args = args
        self.rb_num = 0

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
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train(self):

        localTrainDataLoader = DataLoader(self.local_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=True)
        self.model.cuda().train()
        train_loss = train_acc = train_total = 0
        print(f"client {self.id} training")
        for epoch in tqdm(range(self.args.local_rounds), desc="Epoch"):
            for X, y in localTrainDataLoader:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += y.size(0)

        local_model_paras = self.get_model_parameters()
        time, energy = self.get_cost()
        print(
            f"loss: {train_loss / train_total}  time: {time}  energy: {energy}"
        )
        return local_model_paras, {
            "loss": train_loss / train_total,
            "accuracy": train_acc / train_total,
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

        return (n * B) * np.log2(1 + (p * h * (d**-alpha)) / (N0 * n))

    def get_communication_time(self):
        R_k = self.get_transmission_rate()
        Z_k = self.calculate_model_size()
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
        return C * E * D * (f**2)

    def get_communication_energy(self):
        T_com = self.get_communication_time()
        p = self.attr_dict['transmit_power']
        return T_com * p

    def get_cost(self):
        T = self.get_computation_time() + self.get_communication_time()
        E = self.get_computation_energy() + self.get_communication_energy()
        return T, E


# 初始化客户端,根据一个同长度的属性列表
def init_clients(clients_num, model, optimizer, dataset, attr_dicts, args):
    assert len(attr_dicts
               ) == clients_num, "attr_dicts length must equal to clients_num"
    clients = []
    for i in range(clients_num):
        clients.append(
            Client(i, dataset, optimizer, model, attr_dicts[i], args))
    return clients


# 初始化属性列表 根据高斯分布
def init_attr_dicts(client_num):
    cpu_frequency = np.random.normal(CPU_FREQUENCY_MEAN, CPU_FREQUENCY_STD,
                                     client_num)
    transmit_power = np.random.normal(TRANSMIT_POWER_MEAN, TRANSMIT_POWER_STD,
                                      client_num)
    gain = np.random.normal(GAIN_MEAN, GAIN_STD, client_num)
    distance = np.random.normal(DISTANCE_MEAN, DISTANCE_STD, client_num)

    cpu_frequency = np.clip(cpu_frequency, 1e9, 3e9)
    transmit_power = np.clip(transmit_power, 1e-3, 10e-3)
    gain = np.clip(gain, 1e-9, 1e-7)
    distance = np.clip(distance, 1, 10)

    attr = Attribute(cpu_frequency, transmit_power, gain, distance)
    return [attr.get_attr(i) for i in range(client_num)]


if __name__ == '__main__':
    args = get_args()
    client_num = args.num_clients
    dataset = FedDataset(args)
    attr_dicts = init_attr_dicts(client_num)
    model = resnet18()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    clients = init_clients(client_num, model, optimizer, dataset, attr_dicts,
                           args)
