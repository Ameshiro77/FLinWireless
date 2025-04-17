import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from diffusion import Diffusion
from model import MLP
import numpy as np


class DiffusionAllocor(nn.Module):

    def __init__(self, input_dim, out_dim):
        denoise_net = MLP(state_dim=input_dim, action_dim=out_dim, hidden_dim=256)
        # ===============
        alloctor = Diffusion(
            state_dim=input_dim,
            action_dim=out_dim,
            model=denoise_net,
            max_action=1.,
        )

    def forward(self, x):
        return self.alloctor(x)


class AttenActor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_clients, num_selects=5, constant=10):
        super().__init__()
        self.num_clients = num_clients
        self.input_dim = input_dim

        self.select_encoder = SelectEncoder(input_dim, hidden_dim, num_heads, num_layers)
        self.alloc_encoder = AllocEncoder(input_dim, hidden_dim, 1, num_layers)
        self.select_decoder = SelectDecoder(hidden_dim, num_heads, num_selects=num_selects)
        self.alloc_decoder = AllocDecoder(num_selects * input_dim, num_heads, constant)

    # return:
    # hidden: encoder ouput according to selected_indices
    # log_ll: log(p1p2p3..)=logp(a1|s)+logp(a2|s,a1)...
    # first_dist: first distribution to compute entropy

    def forward(self, x, is_training=False):
        # X : B * (num_clients * input_dim)
        # reshape:
        # print(is_training)
        assert x.shape[1] == self.num_clients * \
            self.input_dim, f"Expected [B, {self.num_clients*self.input_dim}], got {x.shape}"

        # select
        x = x.view(x.shape[0], self.input_dim, self.num_clients).permute(0, 2, 1)  # B *N *Input
        encoder_output = self.select_encoder.forward(x)  # [B, num_clients(SEQ), hidden_dim]
        selected_indices, pi, select_entropy = self.select_decoder(encoder_output, is_training)  # [B, T] (变长)
        logp_select = self._calc_select_likelyhood(pi)

        # alloc
        bs = x.shape[0]
        # selected_hiddens = self._extract_selected_features(x, selected_indices)  # List[[T_i, hidden_dim]]
        selected_hiddens = self.alloc_encoder.forward(x, selected_indices)
        # select_states = self._extract_selected_features(x, selected_indices).view(bs, -1)
        allocation, logp_alloc, alloc_entropy = self.alloc_decoder.forward(selected_hiddens)

        # print(selected_indices, pi, log_ll,joint_entropy)
        #print(selected_indices.shape, allocation.shape)
        act = self._to_act(selected_indices, allocation)

        # print(logp_alloc,logp_select,act)
        # exit()
        log_p = logp_alloc + logp_select
        entropy = select_entropy + alloc_entropy
        return act, None, log_p, entropy

    def _calc_select_likelyhood(self, pi):
        return torch.log(pi).sum(dim=-1)

    def _extract_selected_features(self, x, indices):
        indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        selected_hiddens = torch.gather(x, 1, indices)
        return selected_hiddens

    def _to_act(self, selects, allocs):
        selects = selects.detach().cpu().numpy()
        allocs = allocs.detach().cpu().numpy()
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

            optimizer.zero_grad()
            act, selected_hidden, log_ll, joint_entropy = model.forward(x, is_training=True)

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

    model = AttenActor(input_dim, hidden_dim, num_heads, num_layers, num_clients, num_selects)
    train_actor(model,
                num_clients,
                input_dim,
                num_epochs=10,
                batch_size=4,
                lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu')
