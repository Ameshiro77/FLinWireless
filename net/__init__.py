from net.diffusion import Diffusion
from net.model import MLP, DoubleCritic
import torch
from torch import nn
from torch.nn import functional as F


class ThreeLayerMLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接层
        self.fc3 = nn.Linear(hidden_size, output_size)  # 第三层全连接层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return F.softmax(out, dim=-1)


def choose_actor_critic(state_dim, action_dim, args):
    if args.algo == "diff_sac":
        actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
        actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=actor_net, max_action=100).to(args.device)
        critic = DoubleCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).to(args.device)
    elif args.algo == "mlp_sac":
        actor = ThreeLayerMLP(state_dim, 256, action_dim).to(args.device)
        critic = DoubleCritic(state_dim, action_dim, 256).to(args.device)
    return actor, critic
