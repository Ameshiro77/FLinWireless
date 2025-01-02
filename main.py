import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'env'))
sys.path.append(os.path.join(current_dir, 'net'))
sys.path.append(os.path.join(current_dir, 'policy'))
import torch
import argparse  # 添加 argparse 模块
from gym import spaces
from env.dataset import FedDataset
from env.client import *
from env.model import *
from env.FedEnv import FederatedEnv
import torch.nn.functional as F
from policy.random import RandomPolicy



if __name__ == '__main__':
    # example usage:
    args = get_args()
    dataset = FedDataset(args)
    model = MNISTResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    attr_dicts = init_attr_dicts(args.num_clients)

    # == 初始化客户端数据集
    clients = init_clients(args.num_clients, model, optimizer, dataset,
                           attr_dicts, args)
    for i in range(args.num_clients):
        subset = dataset.get_client_data(i)
        clients[i].local_dataset = subset
        print(f"Client {i} has {len(subset)} samples")

    data_distribution = dataset.get_data_distribution()
    print("Data distribution:")
    for client_id, dist in data_distribution.items():
        print(f"Client {client_id}: {dist},samples num:{sum(dist)}")
    # ==

    env = FederatedEnv(args,
                       dataset,
                       clients,
                       model=model,
                       optimizer=optimizer)
    env.reset()
    policy = RandomPolicy(env.action_space,args)

    print("start training")
    for _ in range(args.global_rounds):
        action = policy.random_action()
        next_state, reward, done, info = env.step(action)
        if done:
            break

    env.close()
