import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet
from diffusion import Diffusion
from model import MLP

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LeakyReLU()
        )

        if len(hidden_dims) > 1:
            for i in range(len(hidden_dims)-1):
                self.model.add_module('linear_{}'.format(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.model.add_module('relu', nn.LeakyReLU())

        self.mean = nn.Linear(hidden_dims[-1], action_space)
        self.log_std = nn.Linear(hidden_dims[-1], action_space)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(0.5)
        self.action_bias = torch.tensor(0.5)
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         (action_space.high - action_space.low) / 2.)
        #     self.action_bias = torch.FloatTensor(
        #         (action_space.high + action_space.low) / 2.)

    def forward(self, state, is_training=True):
        x = self.model(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        y = torch.softmax(z, dim=-1)
        action = y

        log_pi = normal.log_prob(z) - torch.log((1 - y.pow(2)) + EPS)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, None, log_pi, normal.entropy()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DirichletPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, constant=1):
        super(DirichletPolicy, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.constant = constant
        self.apply(weights_init_)

    def forward(self, state, is_training=True):
        alpha_logits = F.softplus(self.policy(state)) + self.constant
        # alpha = F.softmax(alpha_logits, dim=-1) + self.constant
        dist = Dirichlet(alpha_logits)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).unsqueeze(1)
        return actions, None, log_probs, dist.entropy()
        # return actions, log_probs

# continous allocator for DPG-style


class MLPAllocator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.alloc_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.alloc_net(x)
        x = F.softmax(x, dim=-1)
        return x


class DiffusionAllocor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        denoise_net = MLP(state_dim, action_dim, hidden_dim)
        self.net = Diffusion(state_dim, action_dim, model=denoise_net, max_action=1)
    
    def forward(self, x):
        x = self.net(x) # softmax in diffusion
        return x
    
    def loss(self,x,s):
        return self.net.loss(x,s)
