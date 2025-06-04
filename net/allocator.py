import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet
from attention import *

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, 1),
        )

        self.actor_logstd = nn.Parameter(torch.rand(1, action_dim))
        self.apply(weights_init_)

        # self.action_scale = torch.tensor(0.5)
        # self.action_bias = torch.tensor(0.5)

    def forward(self, state, is_training=True):

        action_mean = (torch.tanh(self.mu(state).squeeze(-1)) + 1) / 2
        # action_mean = self.mu(state).squeeze(-1)
        # log_std = self.log_std(x)
        # std = log_std.exp()
        action_logstd = self.actor_logstd.expand_as(action_mean)      # 扩展成和 action_mean 一样形状
        normal = Normal(action_mean, torch.exp(action_logstd))
        # normal = Normal(action_mean, action_logstd)
      
        samples = normal.sample()
        z = (torch.tanh(samples) + 1 ) / 2
        
        log_prob = normal.log_prob(z)
        log_prob = log_prob - torch.log(1 - torch.tanh(z).pow(2) + 1e-7)
        
        #print("logstd", action_logstd)
        #print("samples",samples)
        # z = torch.clip(samples, 0, 1)
        
        
        if is_training:
            # action = torch.softmax(z, dim=-1)
            action = z / torch.sum(z, dim=-1, keepdim=True)
        else:
            # action = torch.softmax(action_mean, dim=-1)
            action = action_mean / torch.sum(action_mean, dim=-1, keepdim=True)
            
        #print("log_pi", log_pi)
        print(action_mean, action_logstd)
        return action, None, log_prob.sum(-1), normal.entropy().sum(-1)


class DirichletPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, constant=1):
        super(DirichletPolicy, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.constant = constant
        self.apply(weights_init_)

    def forward(self, state, is_training=True):
        x = self.policy(state).squeeze(-1)
        print("alloc policy logits:", x)
        # alpha = F.softplus(x) + 1E-3
        alpha = torch.exp(x) + 1e-3
        dist = Dirichlet(alpha)
        print("dirichlet alpha:", alpha)
        if is_training:
            actions = dist.sample()
        else:
            actions = alpha / torch.sum(alpha, dim=-1, keepdim=True)  # dirichlet的期望
        log_probs = dist.log_prob(actions)
        return actions, log_probs, dist.entropy()
        # return actions, log_probs

# continous allocator for DPG-style

class AllocEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, norm_type='layer'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # self.layer = EncoderLayer(input_dim, num_heads)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, norm_type) for _ in range(num_layers)])
        # self.mha = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

    def forward(self, x, select_indices, num_layers=2):
        # x = self.embedding(x)  # h = Wx + b
        selected_x = torch.gather(x, dim=1, index=select_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # (B, K, H)
        x = self.embedding(selected_x)  # h = Wx + b
        for layer in self.layers:
            x = layer(x, x, x)
        return x
        # selected_x = self.embedding(selected_x)  # (B, K, H)
        # x, _ = self.mha.forward(selected_x, selected_x, selected_x)
        # x = x.contiguous().view(x.size(0), -1)  # must flatten
        return x


class AllocDecoder(nn.Module):

    def __init__(self, state_dim, num_choose, num_heads=4):
        super().__init__()
        self.net = DirichletPolicy(state_dim, num_choose, hidden_dim=256)
        # self.net = GaussianPolicy(state_dim, num_choose, hidden_dim=256)

    def forward(self, x, is_training=False):  # x: [B, K, H], 每个 client 的状态
        allocation, log_prob, entropy = self.net.forward(x, is_training)
        return allocation, log_prob, entropy

class MLPAllocator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.alloc_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.alloc_net(x)
        x = F.softmax(x, dim=-1)
        return x


# class DiffusionAllocor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super().__init__()
#         denoise_net = MLP(state_dim, action_dim, hidden_dim)
#         self.net = Diffusion(state_dim, action_dim, model=denoise_net, max_action=1)

#     def forward(self, x):
#         x = self.net(x)  # softmax in diffusion
#         return x

#     def loss(self, x, s):
#         return self.net.loss(x, s)
