import torch
import torch.nn as nn
import torch.nn.functional as F
from allocator import *
from selector import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SelectActor(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, num_heads, num_layers, num_clients, num_selects=5, norm_type='layer',
            decoder_type='mask'):
        super().__init__()
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.num_selects = num_selects

        self.select_encoder = SelectEncoder(input_dim, hidden_dim, num_heads, num_layers, norm_type=norm_type)
        self.select_decoder = SelectDecoder(hidden_dim, num_heads, num_selects=num_selects, decoder_type=decoder_type)

    def forward(self, x, is_training=False):
        if len(x.shape) == 2:  # Reshape input if x: B * (N*S)
            x = x.view(x.shape[0], self.input_dim, self.num_clients).permute(0, 2, 1)  # B * N * Input
        encoder_output = self.select_encoder.forward(x)  # [B, num_clients(SEQ), hidden_dim]
        selected_indices, pi, entropy_select = self.select_decoder(encoder_output, is_training)  # [B, k]
        logp_select = torch.log(pi)  # π(at|s,a1:t-1) 为防止NaN在PPO中clip.

        return selected_indices, logp_select, entropy_select, encoder_output

    def _calc_select_likelihood(self, pi):
        return torch.log(pi).sum(dim=-1)


class AllocActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_selects, norm_type='layer', method='d'):
        super().__init__()
        self.input_dim = input_dim
        self.num_selects = num_selects

        self.alloc_encoder = AllocEncoder(input_dim, hidden_dim, num_heads, num_layers, norm_type)
        self.alloc_decoder = AllocDecoder(hidden_dim, num_selects, None, method=method)

    def forward(self, x, selected_indices, encoder_output=None, is_training=False):
        # Encode and allocate
        selected_hiddens = self.alloc_encoder.forward(x, selected_indices)  # b * k  * h
        allocation, logp_alloc, alloc_entropy = self.alloc_decoder.forward(selected_hiddens, is_training)

        return allocation, logp_alloc, alloc_entropy


class AttenActor(nn.Module):
    def __init__(self, select_actor, alloc_actor, window_size=5, hidden_size=10, lstm=False):
        super().__init__()
        # self.num_clients = select_actor.num_clients
        # self.input_dim = select_actor.input_dim

        self.select_actor = select_actor
        self.alloc_actor = alloc_actor
        self.LSTMProcessor = ObsProcessor(window_size=window_size, hidden_size=hidden_size, lstm=lstm)

    def forward(self, obs_dict, is_training=False):
        x = self.LSTMProcessor(obs_dict)
        # assert x.shape[1] == self.num_clients * self.input_dim, \
        #     f"Expected [B, {self.num_clients*self.input_dim}], got {x.shape}"
        # # Reshape . X:B*(inputdim*N) -> B * N * Input
        # x_reshaped = x.view(x.shape[0], self.input_dim, self.num_clients).permute(0, 2, 1)  # B * N * Input
        # Selection

        selected_indices, logp_select, select_entropy, encoder_output = self.select_actor(x, is_training)
        # Allocation
        allocation, logp_alloc, alloc_entropy = self.alloc_actor(x, selected_indices, is_training)

        # Combine results
        act = self._to_act(selected_indices, allocation)
        # log_p = logp_alloc + logp_select

        log_p = torch.cat([logp_select, logp_alloc.unsqueeze(-1)], dim=-1)  # [B, k + 1]

        # entropy = select_entropy + alloc_entropy
        print(select_entropy.shape, alloc_entropy.shape)
        print(select_entropy, alloc_entropy)
        # exit()
        return act, log_p, select_entropy + alloc_entropy

    def _to_act(self, selects, allocs):
        selects = selects.detach().cpu().numpy()
        allocs = allocs.detach().cpu().numpy()
        # print(selects, allocs)
        return np.stack([selects, allocs], axis=1)


def train_actor(model, num_clients, input_dim, num_epochs=10, batch_size=2, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    data = torch.randn(100, num_clients * input_dim)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, ) in enumerate(dataloader):
            x = x.to(device)
            dict = {'obs': x}  # 模拟输入字典
            optimizer.zero_grad()
            act, selected_hidden, log_ll, joint_entropy = model.forward(dict, is_training=True)
            # 损失函数，这里简单使用 -log_likelihood + joint_entropy 作为训练目标
            loss = (-log_ll + 0.01 * joint_entropy).mean()
            print(loss)

            loss.backward()

            # 梯度检查
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"⚠️ No gradient for {name}")
            #     elif torch.isnan(param.grad).any():
            #         print(f"❌ NaN gradient in {name}")
            #     else:
            #         print(f"✅ {name}: grad mean {param.grad.mean():.4f}, max {param.grad.abs().max():.4f}")
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            print(f"Joint entropy: {joint_entropy.mean().item():.4f}")

        scheduler.step()


if __name__ == "__main__":
    # 参数设置
    num_clients = 10
    input_dim = 4
    hidden_dim = 8
    num_heads = 4
    num_layers = 3
    num_selects = 5

    select_actor = SelectActor(input_dim, hidden_dim, num_heads, num_layers, num_clients, num_selects)
    alloc_actor = AllocActor(input_dim, hidden_dim, num_heads, num_layers, num_selects)
    model = AttenActor(select_actor, alloc_actor)
    train_actor(model,
                num_clients,
                input_dim,
                num_epochs=10,
                batch_size=4,
                lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu')
