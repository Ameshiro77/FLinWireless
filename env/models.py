import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * 4 * 4, 10)  # For 28x28 input: ((28-4)/2-4)/2 = 4x4

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 24x24 -> 12x12
        x = self.pool(F.relu(self.conv2(x)))  # 12x12 -> 8x8 -> 4x4
        x = x.view(-1, 8 * 4 * 4)
        x = self.fc(x)
        return x

class FashionMNIST_LeNet5(nn.Module):
    def __init__(self):
        super(FashionMNIST_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # For 28x28 input: ((28-4)/2-4)/2 = 4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 24x24 -> 12x12
        x = self.pool(F.relu(self.conv2(x)))  # 12x12 -> 8x8 -> 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10_LeNet5(nn.Module):
    def __init__(self):
        super(CIFAR10_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # For 32x32 input: ((32-4)/2-4)/2 = 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 10x10 -> 5x5
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def choose_model(args):
    dataset = args.dataset
    if dataset == "MNIST":
        return MNIST_CNN()
    elif dataset == "Fashion":
        return FashionMNIST_LeNet5()
    elif dataset == "CIFAR10":
        return CIFAR10_LeNet5()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")