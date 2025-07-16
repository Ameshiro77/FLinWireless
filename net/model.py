import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tianshou.data import Batch
import numpy as np


# 用于特征预处理。特征来自环境的obs，通常是np。
# 只处理需要LSTM处理的时序数据。
class ObsProcessor(nn.Module):
    def __init__(self, window_size, hidden_size, lstm=False):
        super().__init__()
        self.use_lstm = lstm
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.window_size = window_size
        self.hidden_size = hidden_size

    def forward(self, obs):
        batch_size = None
        clients_num = None
        features_list = []
        keys = sorted(obs.keys())  # Sort keys to ensure consistent order！
        for key in keys:
            value = obs[key]
            # print(key)
            # print(value.shape)
            # value is a numpy array, convert it to a tensor
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float().cuda()  # Convert to tensor and ensure it's float

            # value: shape (batch_size, clients_num, something)

            if batch_size is None:
                batch_size, clients_num = value.shape[0], value.shape[1]

            if value.ndim == 3 and value.shape[-1] == self.window_size:
                if self.use_lstm:
                    # Historical data, apply LSTM
                    seq = value.view(batch_size * clients_num, self.window_size, 1)
                    out, (h_n, c_n) = self.lstm(seq)
                    feature = h_n[-1]
                    feature = feature.view(batch_size, clients_num, self.hidden_size)
                    features_list.append(feature)
                else:
                    continue  # Skip if LSTM is not used

            else:
                # Other features, directly append
                feature = value.unsqueeze(-1)  # (batch_size, clients_num, feature_dim)
                features_list.append(feature)

        # Concatenate all features along the last dimension
        # for feature in features_list:
        #     print(feature.shape)
        final_feature = torch.cat(features_list, dim=-1)  # (batch_size, clients_num, total_feature_dim)
        # print(final_feature)
        # exit()
        return final_feature


class Critic_V(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, window_size=5, hidden_size=10, activation="mish", LSTM=False):
        super().__init__()
        _act = nn.Mish if activation == "mish" else nn.LeakyReLU
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1),
        )
        self.LSTMProcessor = ObsProcessor(window_size=window_size, hidden_size=hidden_size, lstm=LSTM)

    def forward(self, obs):
        x = self.LSTMProcessor(obs)
        x = x.reshape(x.size(0), -1)
        return self.q_net(x)


class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super().__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1),
        )
        self.LSTMProcessor = ObsProcessor(window_size=5, hidden_size=10)

    def forward(self, obs, act):
        obs = self.LSTMProcessor(obs)
        obs = obs.reshape(obs.size(0), -1)
        act = act.reshape(act.size(0), -1)
        x = torch.cat([obs, act], dim=1)
        return self.q_net(x)


class Critic_Attn_V(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, activation="ReLU"):
        super().__init__()
        _act = nn.Mish if activation == "mish" else nn.LeakyReLU
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        obs = obs.reshape(obs.size(0), -1)
        return self.q_net(obs)


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU
        self.action_dim = action_dim
        self.LSTMProcessor = ObsProcessor(window_size=5, hidden_size=10)

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
        obs = self.LSTMProcessor(obs)
        obs = obs.reshape(obs.size(0), -1)
        act = act.reshape(act.size(0), -1)
        x = torch.cat([obs, act], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, act):
        return torch.min(*self.forward(obs, act))

# class MLP(nn.Module):
#     def __init__(
#         self, state_dim, action_dim, hidden_dim=256, t_dim=16, activation="mish"
#     ):
#         super(MLP, self).__init__()
#         _act = nn.Mish if activation == "mish" else nn.ReLU
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(t_dim),
#             nn.Linear(t_dim, t_dim * 2),
#             _act(),
#             nn.Linear(t_dim * 2, t_dim),
#         )
#         self.mid_layer = nn.Sequential(
#             nn.Linear(state_dim + action_dim + t_dim, hidden_dim),
#             _act(),
#             nn.Linear(hidden_dim, hidden_dim),
#             _act(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#         self.final_layer = nn.Tanh()

#     def forward(self, x, time, state):  # x:action
#         t = self.time_mlp(time)
#         state = state.reshape(state.size(0), -1)
#         x = torch.cat([x, t, state], dim=1)  # 16(t) + action(x) + state
#         x = self.mid_layer(x)
#         return self.final_layer(x)
