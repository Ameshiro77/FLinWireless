from env.dataset import FedDataset
from env.client import *
from env.models import *
from env.FedEnv import FederatedEnv, make_env


def gen_env(args):
    dataset = FedDataset(args)
    if args.dataset == 'MNIST':
        model = MNISTResNet()
    else:
        raise ValueError("no dataset!")
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
    print("clients attr:")
    for i in range(args.num_clients):
        print(f"client {i} initialized,attr:{attr_dicts[i]}")
    # ==

    env, train_envs, test_envs = make_env(args, dataset, clients, model=model)
    return env, train_envs, test_envs
