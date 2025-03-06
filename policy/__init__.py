from policy.rand import RandomPolicy
from policy.difftd3 import DiffusionTD3
from policy.diffpolicy import DiffusionSAC
import torch


def choose_policy(actor, actor_optim, critic, critic_optim, args):
    if args.algo == 'diff_sac':
        policy = DiffusionSAC(actor,
                              actor_optim,
                              args.num_clients,
                              critic,
                              critic_optim,
                              dist_fn=torch.distributions.Categorical,
                              device=args.device,
                              gamma=0.95,
                              estimation_step=3,
                              is_not_alloc=args.no_allocation,
                              )
    elif args.algo == 'rand':
        policy = RandomPolicy
    elif args.algo == 'diff_td3':
        policy = DiffusionTD3(actor,
                              actor_optim,
                              critic=critic,
                              critic_optim=critic_optim,
                              device=args.device,
                              tau=0.005,
                              gamma=0.99,
                              training_noise=0.1,
                              policy_noise=0.2,
                              noise_clip=0.5
                              )
    else:
        raise ValueError('Unknown policy!supported:diff_sac,rand,diff')
    return policy
