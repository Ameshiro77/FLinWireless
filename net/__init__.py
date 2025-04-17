from net.diffusion import Diffusion
from net.model import MLP as NoiseMLP
from net.model import DoubleCritic
import torch
from torch import nn
from torch.nn import functional as F
from tianshou.utils.net.common import MLP as SimpleMLP
from net.db_actor import *
from net.atten_actor import *


def choose_actor_critic(state_dim, action_dim, args):

    # actor = DbranchActor(state_dim, action_dim, args).to(args.device)
    actor = AttenActor(7, 128, 4, 3, 10, args.top_k, args.constant).to(args.device)
    # critic = DoubleCritic(state_dim, action_dim, 256).to(args.device)
    critic = Critic_V(state_dim, 256).to(args.device)
    return actor, critic
