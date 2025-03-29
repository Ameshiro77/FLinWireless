import torch
import torch.nn as nn
from helpers import SinusoidalPosEmb
import math

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
        x = torch.cat([x, t, state], dim=1)  # 16(t) + action(x) + state
        
        x = self.mid_layer(x)
        return self.final_layer(x)


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU
        self.action_dim = action_dim

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

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # 一般来说，
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim
        assert self.head_dim * nums_head == hidden_dim
        # 一般默认有 bias，需要时刻主意，hidden_dim = head_dim * nums_head，所以最终是可以算成是 n 个矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # X shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)
        batch_size, seq_len, _ = q.size()
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)
        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        # 用 head_dim，而不是 hidden_dim
        # bs , nums_head, seq_len, seq_len
        attention_weight = (q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim))

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 变成 [batch_size, 1, 1, num_clients + 1]
            mask = mask.expand(-1, 4, seq_len, -1)  # 变成 [batch_size, num_heads, num_clients + 1, num_clients + 1]

            # print(attention_weight.shape)
            # print(mask.shape)
            # exit()
            attention_weight = attention_weight.masked_fill(mask == 0, float("-inf"))

        # 第四个维度 softmax
        attention_weight = torch.softmax(attention_weight, dim=3)
        # print(attention_weight.shape)

        output_mid = attention_weight @ v_state

        # reshape to (batch, seq_len, num_head, head_dim)
        output_mid = output_mid.transpose(1, 2).contiguous()
        #  (batch, seq, hidden_dim),
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        # print(output.shape)
        return output
