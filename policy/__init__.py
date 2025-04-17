from policy.TD3 import TD3Policy
from policy.diffpolicy import DiffusionSAC
from policy.TD3BC import TD3BCPolicy
from policy.sac import DiffusionSAC2
from policy.pg import PGPolicy
from policy.PPO import PPOPolicy
from policy.ddpg import DDPGPolicy
import torch


def choose_policy(actor, actor_optim, critic, critic_optim, args):
    algo = args.algo.split('_')[-1]
    if algo == 'pg':
        policy = PGPolicy(
            actor,
            actor_optim,
            device=args.device,
            dist_fn=None
        )
    elif algo == 'ppo':
        policy = PPOPolicy(
            actor,
            critic,
            optim_lr=1e-4,
            dist_fn=torch.distributions.Categorical,
            device=args.device,
        )
    return policy
    algo = args.algo.split('_')[-1]
    if algo == 'sac':
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
                              task=args.task
                              )
    elif algo == 'sac2':
        policy = DiffusionSAC2(actor,
                               actor_optim,
                               args.num_clients,
                               critic,
                               critic_optim,
                               dist_fn=torch.distributions.Categorical,
                               device=args.device,
                               gamma=0.95,
                               estimation_step=3,
                               is_not_alloc=args.no_allocation,
                               task=args.task
                               )
    elif algo == 'td3':
        policy = TD3Policy(actor,
                           actor_optim,
                           critic,
                           critic_optim=critic_optim,
                           device=args.device,
                           tau=0.005,
                           gamma=0.99,
                           policy_noise=0.2,
                           noise_clip=0.5,
                           alpha=2.5,
                           task=args.task
                           )
    elif algo == 'td3bc':
        policy = TD3BCPolicy(actor,
                             actor_optim,
                             critic,
                             critic_optim,
                             device=args.device,
                             tau=0.005,
                             gamma=0.99,
                             task=args.task
                             )
    else:
        raise ValueError('Unknown policy!supported:sac,rand,td3bc')
    return policy
