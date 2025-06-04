from net.model import DoubleCritic
import torch
from torch import nn
from torch.nn import functional as F
from tianshou.utils.net.common import MLP as SimpleMLP
from net.atten_actor import *


class SingleSelector(nn.Module):
    def __init__(self, window_size, hidden_size, selector):
        super().__init__()
        self.selector = selector
        self.LSTMProcessor = ObsProcessor(window_size=window_size, hidden_size=hidden_size)

    def forward(self, obs_dict, is_training=False):
        x = self.LSTMProcessor(obs_dict)
        selected_indices, logp_select, entropy_select, encoder_output = self.selector(x, is_training)
        return selected_indices, logp_select, entropy_select


def choose_actor_critic(state_dim, num_selects, args):
    num_clients, num_selects = args.num_clients, num_selects
    window_size, hidden_size = args.window_size, args.hidden_size  # LSTM
    input_dim = state_dim // num_clients

    if args.task == "hybrid":
        # selector
        select_hidden_dim, select_num_heads, select_num_layers = args.select_hidden_dim, args.select_num_heads, args.select_num_layers
        select_norm_type = args.select_norm_type
        select_decoder_type = args.select_decoder
        select_actor = SelectActor(
            input_dim, select_hidden_dim, select_num_heads, select_num_layers, num_clients, num_selects,
            norm_type=select_norm_type, decoder_type=select_decoder_type).to(
            args.device)

        # allocator
        alloc_hidden_dim, alloc_num_heads, alloc_num_layers = args.alloc_hidden_dim, args.alloc_num_heads, args.alloc_num_layers
        alloc_norm_type = args.alloc_norm_type
        alloc_actor = AllocActor(input_dim, alloc_hidden_dim, alloc_num_heads, alloc_num_layers,
                                 num_selects, alloc_norm_type).to(args.device)

        actor = AttenActor(select_actor, alloc_actor, window_size, hidden_size).to(args.device)

        # critic = DoubleCritic(state_dim, action_dim, 256).to(args.device)
        critic = Critic_V(state_dim, 256, window_size, hidden_size).to(args.device)
        return actor, critic

    elif args.task == "acc":
        # selector
        select_hidden_dim, select_num_heads, select_num_layers = args.select_hidden_dim, args.select_num_heads, args.select_num_layers
        select_norm_type = args.select_norm_type
        select_decoder_type = args.select_decoder
        select_actor = SelectActor(
            input_dim, select_hidden_dim, select_num_heads, select_num_layers, num_clients, num_selects,
            norm_type=select_norm_type, decoder_type=select_decoder_type).to(
            args.device)
        actor = SingleSelector(window_size, hidden_size, select_actor).to(args.device)
        critic = Critic_V(state_dim, 256, window_size, hidden_size).to(args.device)

        return actor, critic

    else:
        return None, None
