import argparse
from types import SimpleNamespace
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, Subset
import random
from config import get_args
from fedlab.utils.dataset.partition import CIFAR10Partitioner, MNISTPartitioner, FMNISTPartitioner
from fedlab.utils.functional import partition_report
"""
管理联邦学习过程的数据集分配。
"""


class FedDataset(Dataset):
    # 构建不需要client类，但是需要知道num
    def __init__(self, args):

        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.dataset_name = args.dataset
        self.num_clients = args.num_clients  # 客户端数量。根据id分配。
        self.alpha = args.dir_alpha  # noniid分布因子

        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.non_iid = args.non_iid

        # ===
        self.train_data, self.test_data = self.load_dataset()  # 得到全部数据集
        self.train_labels = np.array(self.train_data.targets)

        valid_indices = random.sample(range(len(self.test_data)), len(self.test_data) // 10)
        self.valid_data = Subset(self.test_data, valid_indices)

        if self.alpha == 0.0:
            self.method = 'iid'
        else:
            self.method = 'dirichlet'
        self.partition = self.get_federated_partition(self.dataset_name, self.train_data.targets)
        self.client_data_indices = self.partition.client_dict

        # if self.non_iid:  # 划分
        #     self.client_data_indices = self.non_iid_split()
        # else:
        #     self.client_data_indices = self.iid_split()

    def get_federated_partition(self, dataset_name, targets):
        # FedML为CIFAR和MNIST提供的接口不同
        # ref:https://github.com/SMILELab-FL/FedLab-benchmarks/blob/master/fedlab_benchmarks/datasets/fmnist/fmnist_tutorial.ipynb
        if self.method == 'dirichlet':
            mnist_meth = 'noniid-labeldir'
            if dataset_name == 'CIFAR10':
                partitioner = CIFAR10Partitioner(
                    targets=targets,
                    num_clients=self.num_clients,
                    balance=None,
                    partition=self.method,  # dirichlet | shards
                    num_shards=2 * self.num_clients,
                    dir_alpha=self.alpha,
                    seed=self.seed,
                )
            elif dataset_name == 'Fashion':
                partitioner = FMNISTPartitioner(
                    targets=targets,
                    num_clients=self.num_clients,
                    partition=mnist_meth,
                    dir_alpha=self.alpha,
                    major_classes_num=2,
                    seed=self.seed,
                )
            elif dataset_name == 'MNIST':
                partitioner = MNISTPartitioner(
                    targets=targets,
                    num_clients=self.num_clients,
                    partition=mnist_meth,
                    dir_alpha=self.alpha,
                    major_classes_num=2,
                    seed=self.seed,
                )
        elif self.method == 'iid':
            if dataset_name == 'CIFAR10':
                partitioner = CIFAR10Partitioner(
                    targets=targets,
                    num_clients=self.num_clients,
                    balance=False,
                    partition=self.method,
                    unbalance_sgm=0.3,
                    seed=self.seed,
                )
            elif dataset_name == 'Fashion':
                # partitioner = FMNISTPartitioner(
                #     targets=targets,
                #     num_clients=self.num_clients,
                #     partition="iid", 
                #     seed=self.seed,
                # )
                partitioner = self.iid_split()
            elif dataset_name == 'MNIST':
                # partitioner = MNISTPartitioner(
                #     targets=targets,
                #     num_clients=self.num_clients,
                #     partition='iid',
                #     seed=self.seed,
                # )
                partitioner = self.iid_split()
        else:
            raise ValueError("Unsupported dataset. Choose from 'MNIST', 'CIFAR10', or 'Fashion'.")
        csv_file = f"./data/dist/{dataset_name}_{self.alpha}.csv"
        partition_report(
            targets,
            partitioner.client_dict,
            class_num=10,  # 都是10
            verbose=False,
            file=csv_file)
        return partitioner

    def load_dataset(self):
        transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        transform_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if self.dataset_name == 'MNIST':
            train_data = torchvision.datasets.MNIST(root=self.data_dir,
                                                    train=True,
                                                    download=True,
                                                    transform=transform_mnist)
            test_data = torchvision.datasets.MNIST(root=self.data_dir,
                                                   train=False,
                                                   download=True,
                                                   transform=transform_mnist)
        elif self.dataset_name == 'Fashion':
            train_data = torchvision.datasets.FashionMNIST(root=self.data_dir,
                                                           train=True,
                                                           download=True,
                                                           transform=transform_mnist)
            test_data = torchvision.datasets.FashionMNIST(root=self.data_dir,
                                                          train=False,
                                                          download=True,
                                                          transform=transform_mnist)
        elif self.dataset_name == 'CIFAR10':
            train_data = torchvision.datasets.CIFAR10(root=self.data_dir,
                                                      train=True,
                                                      download=True,
                                                      transform=transform_cifar)
            test_data = torchvision.datasets.CIFAR10(root=self.data_dir,
                                                     train=False,
                                                     download=True,
                                                     transform=transform_cifar)
        else:
            raise ValueError("Unsupported dataset. Choose from 'MNIST', 'CIFAR10', or 'Fashion'.")

        return train_data, test_data

    def iid_split(self):
        indices = np.arange(len(self.train_data))
        np.random.shuffle(indices)

        # 设置数量不均但分布 IID
        unbalance_sgm = 0.3
        mean_size = len(indices) / self.num_clients
        client_sample_counts = np.random.normal(loc=mean_size, scale=unbalance_sgm * mean_size, size=self.num_clients)
        client_sample_counts = np.clip(client_sample_counts, a_min=10, a_max=None)
        client_sample_counts = (client_sample_counts / np.sum(client_sample_counts) * len(indices)).astype(int)

        # 调整总量误差
        diff = len(indices) - np.sum(client_sample_counts)
        for i in range(abs(diff)):
            client_sample_counts[i % self.num_clients] += np.sign(diff)

        client_dict = {}
        start = 0
        for i, count in enumerate(client_sample_counts):
            client_dict[i] = indices[start:start + count].tolist()
            start += count

        return SimpleNamespace(client_dict=client_dict)

    # 根据id返回给定客户端的数据
    def get_client_data(self, client_id):
        # Get data indices for the specified client
        assert client_id in self.client_data_indices, "Invalid client ID!"
        indices = self.client_data_indices[client_id]
        client_subset = Subset(self.train_data, indices)
        return client_subset

    def get_test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)\


    def get_data_distribution(self):
        # Analyze data distribution for each client
        targets = np.array(self.train_data.targets)
        num_classes = len(np.unique(targets))

        distribution = {}
        for client_id, indices in self.client_data_indices.items():
            class_counts = np.zeros(num_classes, dtype=int)
            for idx in indices:
                class_counts[targets[idx]] += 1
            distribution[client_id] = class_counts.tolist()

        return distribution

    # 获取客户端数据总数
    def get_client_data_sizes(self):
        client_sizes = {client_id: len(indices) for client_id, indices in self.client_data_indices.items()}
        return client_sizes

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index]


if __name__ == "__main__":
    args = get_args()

    federated_dataset = FedDataset(args)

    print(federated_dataset.get_client_data_sizes())

    data_distribution = federated_dataset.get_data_distribution()
    for client_id, dist in data_distribution.items():
        print(f"Client {client_id}: {dist}")

    # client_subset = federated_dataset.get_client_data(client_id=0)
    # client_loader = DataLoader(client_subset, batch_size=args.batch_size, shuffle=True)

    # for batch_idx, (data, target) in enumerate(client_loader):
    #     print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
    #     break
