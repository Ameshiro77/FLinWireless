import sys  # noqa
import os  # noqa

current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa

from env import *
from policy import *
from net import *
import json
from datetime import datetime

# 'data_sizes': [5632,6699,4489,2845,6991,9946,6402,6714,4093,6140]

from env.FedEnv import minmax_normalize, sum_normalize
import random


import random
from tianshou.policy import BasePolicy
from env.config import *
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch
import numpy as np



        

if __name__ == "__main__":

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_choose = args.num_choose
    state_dim = (args.input_dim + args.hidden_size) * args.num_clients
    
    # env, _, _ = gen_env(args)
    env = make_test_env(args)
    env.is_fed_train = True

    policy = choose_actor_critic(state_dim,num_choose,args)
    dataset = args.dataset
    dir_coef = str(args.dir_alpha)
    num_clients = str(args.num_clients)

    exp_dir = "results/" + f"acc_test/" + dataset
    os.makedirs(exp_dir, exist_ok=True)

    print("======== evaluate =======")
    obs, info = env.reset()  # 重置环境，返回初始观测值
    result_dict = {
        'current_round': [],
        'global_loss': [],
        'global_accuracy': [],
        'total_time': [],
        'total_energy': [],
    }
    # from tianshou.policy
    done = False
    while not done:
        action = policy.forward(obs, info=info).act
        obs, rew, done, done, info = env.step(action)
        for key, value in info.items():
            if key in result_dict:
                result_dict[key].append(value)
        if done:
            break
    env.close()

    print(result_dict)
    with open(os.path.join(exp_dir, f"{num_clients}_{dir_coef}_result.json"), "w") as f:
        json.dump(result_dict, f, indent=4)
