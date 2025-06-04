from copy import deepcopy
from env.dataset import FedDataset
from env.client import *
from env.models import *
from env.FedEnv import FederatedEnv
from config import get_args
from tianshou.env import DummyVectorEnv

def make_test_env(args):
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
    print("clients attr:")
    for i in range(args.num_clients):
        print(f"client {i} initialized,attr:{attr_dicts[i]}")
    env = FederatedEnv(args, dataset, clients, model)
    return env


def gen_env(args):
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
    print("clients attr:")
    for i in range(args.num_clients):
        print(f"client {i} initialized,attr:{attr_dicts[i]}")

    # ==
    # 必须包装成tianshou可以识别的向量环境。
    def make_env():
        # 每个环境应该有自己独立的实例
        env = FederatedEnv(args, dataset, deepcopy(clients), deepcopy(model))
        return env
    
    env = FederatedEnv(args, dataset, deepcopy(clients), deepcopy(model))
    train_envs = DummyVectorEnv([lambda: make_env() for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(args.test_num)])
    
    return env, train_envs, test_envs

