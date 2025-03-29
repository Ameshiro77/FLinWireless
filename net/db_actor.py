from net.diffusion import Diffusion
from net.model import MLP as NoiseMLP
from net.model import DoubleCritic
import torch
from torch import nn
from torch.nn import functional as F
from tianshou.utils.net.common import MLP as SimpleMLP


class ThreeLayerMLP(torch.nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, activation="mish"):
        super().__init__()
        act_fn = nn.Mish if activation == "mish" else nn.ReLU
        self.mid_layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), act_fn(), nn.Linear(hidden_dim, hidden_dim),
                                       act_fn(), nn.Linear(hidden_dim, action_dim), nn.LeakyReLU())

    def forward(self, x):
        logits = self.mid_layer(x)
        return F.softmax(logits, dim=-1)


class DbranchActor(torch.nn.Module):

    def __init__(self, state_dim, action_dim, args, activation="mish"):
        super().__init__()
        act_fn = nn.Mish if activation == "mish" else nn.ReLU
        self.args = args
        self.threshold = args.threshold
        self.dbranch = args.dbranch
        # === output probs === #
        if args.algo.startswith("diff"):
            noise_net = NoiseMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
            self.feature_extractor = Diffusion(state_dim=state_dim,
                                               action_dim=action_dim,
                                               model=noise_net,
                                               max_action=100).to(args.device)
        elif args.algo.startswith("mlp"):
            self.feature_extractor = SimpleMLP(state_dim, action_dim, (64, 64), activation=act_fn)
        if self.dbranch:
            self.probs_net = SimpleMLP(action_dim, action_dim, (128, 128))
            self.alloc_net = SimpleMLP(action_dim, action_dim, (128, 128))
            self.probs = None
            self.allocs = None

    def forward(self, state, exploration_fn=None):
        x = self.feature_extractor(state)
        if self.args.task == 'gym':
            return x
        if not self.dbranch:
            x = torch.softmax(x, dim=-1)
            return x, x, x
        else:
            # selection probs branch
            probs = self.probs_net(x)
            probs = torch.sigmoid(probs)  # 输出选择概率 (0 ~ 1)
            self.probs = probs

            # 使用 STE 保持梯度回传
            binary_selection = (probs - probs.detach() + torch.round(probs))

            # bandwidth allocation branch
            allocs = self.alloc_net(x)
            origin_allocs = torch.softmax(allocs, dim=-1).tolist()
            if exploration_fn:
                allocs = exploration_fn(allocs)
            allocs = torch.softmax(allocs, dim=-1)  # 分配带宽 (0 ~ 1)
            self.allocs = allocs

            result = torch.mul(binary_selection, allocs)
            action = result / (result.sum(dim=-1, keepdim=True) + 1e-8)
            print(f"probs:{probs}\noriginal_allocs:{origin_allocs}\nallocs:{allocs}\naction:{action}")
            return probs, allocs, action

    def compute_loss(self, td3_loss):
        lambda_1, lambda_2 = self.args.lambda_1, self.args.lambda_2
        # === TD3 损失 ===
        loss_td3 = td3_loss

        # === 分支对齐损失 ===
        loss_align = F.kl_div(self.allocs.log(), self.probs, reduction='batchmean')

        #=== 概率增强损失 ===
        loss_boost = -(1 - self.probs) * loss_td3.detach()
        loss_boost = loss_boost.mean()

        # === 总损失 ===
        loss = loss_td3 + lambda_1 * loss_align + lambda_2 * loss_boost
        return loss, {'loss_pi': loss_td3.item(), 'loss_align': loss_align.item(), 'loss_boost': loss_boost.item()}
