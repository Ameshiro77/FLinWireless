import torch
import torch.nn as nn
from helpers import SinusoidalPosEmb


class MLP(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, t_dim=16, activation="mish"
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):  # x:action
        t = self.time_mlp(time)
        state = state.reshape(state.size(0), -1)
        x = torch.cat([x, t, state], dim=1)  # 16(t) + action + state
        x = self.mid_layer(x)
        return self.final_layer(x)


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        obs = obs.reshape(obs.size(0), -1)
        act = act.reshape(act.size(0), -1)
        x = torch.cat([obs, act], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, act):
        obs = obs.reshape(obs.size(0), -1)
        act = act.reshape(act.size(0), -1)
        return torch.min(*self.forward(obs, act))
