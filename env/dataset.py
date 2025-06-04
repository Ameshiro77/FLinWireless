import argparse
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
        self.alpha = args.dir_alpha  #noniid分布因子
        if self.alpha == 0:
            self.method = 'shards'
        else:
            self.method = 'dirichlet'
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.non_iid = args.non_iid

        # ===
        self.train_data, self.test_data = self.load_dataset()  # 得到全部数据集
        self.train_labels = np.array(self.train_data.targets)

        valid_indices = random.sample(range(len(self.test_data)), len(self.test_data) // 10)
        self.valid_data = Subset(self.test_data, valid_indices)

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
        elif self.method == 'shards':
            mnist_meth = "noniid-#label"

        if dataset_name == 'CIFAR10':
            partitioner = CIFAR10Partitioner(
                targets=targets,
                num_clients=self.num_clients,
                balance=None,
                partition=self.method,  ## dirichlet | shards
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
        else:
            raise ValueError("Unsupported dataset. Choose from 'MNIST', 'CIFAR10', or 'Fashion'.")
        csv_file = f"./data/dist/{dataset_name}_{self.method}.csv"
        partition_report(
            targets,
            partitioner.client_dict,
            class_num=10,  #都是10
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

    def non_iid_split(self):
        # Get labels for all training samples
        targets = np.array(self.train_data.targets)
        num_classes = len(np.unique(targets))

        # Dirichlet distribution to simulate non-iid
        client_data_indices = {i: [] for i in range(self.num_clients)}
        for c in range(num_classes):
            # Get indices for all data points with class c
            class_indices = np.where(targets == c)[0]
            np.random.shuffle(class_indices)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(class_indices)).astype(int)

            # Assign indices to clients
            start_idx = 0
            for i in range(self.num_clients):
                client_data_indices[i].extend(class_indices[start_idx:start_idx + proportions[i]])
                start_idx += proportions[i]

        return client_data_indices

    def iid_split(self):
        indices = np.arange(len(self.train_data))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, self.num_clients)
        client_data_indices = {i: split.tolist() for i, split in enumerate(split_indices)}

        return client_data_indices

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

    # print(federated_dataset.get_client_data_sizes())

    # data_distribution = federated_dataset.get_data_distribution()
    # for client_id, dist in data_distribution.items():
    #     print(f"Client {client_id}: {dist}")

    # client_subset = federated_dataset.get_client_data(client_id=0)
    # client_loader = DataLoader(client_subset, batch_size=args.batch_size, shuffle=True)

    # for batch_idx, (data, target) in enumerate(client_loader):
    #     print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
    #     break
