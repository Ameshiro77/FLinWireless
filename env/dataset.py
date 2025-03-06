import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, Subset

from config import get_args

"""
管理联邦学习过程的数据集分配。
"""


class FedDataset(Dataset):
    # 构建不需要client类，但是需要知道num
    def __init__(self, args):

        self.seed = args.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.dataset_name = args.dataset
        self.num_clients = args.num_clients  # 客户端数量。根据id分配。
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.non_iid = args.non_iid
        self.train_data, self.test_data = self.load_dataset()  # 得到全部数据集
        if self.non_iid:  # 划分
            self.client_data_indices = self.non_iid_split()
        else:
            self.client_data_indices = self.iid_split()

    def load_dataset(self):
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if self.dataset_name == 'MNIST':
            train_data = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=transform_mnist
            )
            test_data = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_mnist
            )
        elif self.dataset_name == 'CIFAR10':
            train_data = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform_cifar
            )
            test_data = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=transform_cifar
            )
        elif self.dataset_name == 'CIFAR100':
            train_data = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=transform_cifar
            )
            test_data = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=transform_cifar
            )
        else:
            raise ValueError(
                "Unsupported dataset. Choose from 'MNIST', 'CIFAR10', or 'CIFAR100'.")

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
                client_data_indices[i].extend(
                    class_indices[start_idx:start_idx + proportions[i]])
                start_idx += proportions[i]

        return client_data_indices

    def iid_split(self):
        indices = np.arange(len(self.train_data))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, self.num_clients)
        client_data_indices = {i: split.tolist()
                               for i, split in enumerate(split_indices)}

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
        client_sizes = {client_id: len(
            indices) for client_id, indices in self.client_data_indices.items()}
        return client_sizes
    
    # 重写dataset用的
    def __len__(self):
        # 返回数据集的长度
        return len(self.train_data)

    def __getitem__(self, index):
        # 返回指定索引的数据和标签
        return self.train_data[index]


if __name__ == "__main__":
    args = get_args()

    federated_dataset = FedDataset(args)
    client_loader = federated_dataset.get_client_data(client_id=0)
    test_loader = federated_dataset.get_test_dataloader()
    print(federated_dataset.get_client_data_sizes())

    data_distribution = federated_dataset.get_data_distribution()
    for client_id, dist in data_distribution.items():
        print(f"Client {client_id}: {dist}")
        
    for batch_idx, (data, target) in enumerate(client_loader):
        print(
            f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        break
