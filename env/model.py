from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.nn import Module, Conv2d, Linear, MaxPool2d
import math
import torch.nn as nn
import copy
import torch
import torch.nn.functional as F


class MNISTResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet, self).__init__()
        # 加载 ResNet18 作为基础模型
        self.base_model = resnet18(pretrained=False)
        
        # 修改输入层以适配 MNIST (单通道)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改输出层以适配 num_classes
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

