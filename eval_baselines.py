import sys  # noqa
import os  # noqa

current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa

import torch
from gym import spaces
import torch.nn.functional as F
from tianshou.policy import DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer
from tianshou.data import VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from tools import *
from env import *
from policy import *
from net import *
from baselines import *
from datetime import datetime
from tianshou.data import Batch
import json

if __name__ == "__main__":

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.update_gain = True
    # env, _, _ = gen_env(args)
    env = make_test_env(args)
    env.is_fed_train = True

    dataset = args.dataset
    fcmt = {
        'MNIST': 0.04,
        'Fashion': 0.3,
        'CIFAR10': 0.3,
    }
    min_time = fcmt[dataset]

    fedavg_policy = (FedAvgPolicy(args), 'fedavg')
    greedy_policy = (GreedyPolicy(args), 'greedy')
    fedcs_policy = (FedCSPolicy(args, min_time), 'fedcs')
    policies = [fedavg_policy, greedy_policy, fedcs_policy]
    # policies = [greedy_policy]
    # policies = [fedcs_policy]

    dataset = args.dataset
    dir_coef = str(args.dir_alpha)
    num_clients = str(args.num_clients)
    lc_rds = str(args.local_rounds)
        
    exp_dir = "results/" + f"{args.dataset}/" + f"alpha={dir_coef}/" + f"num={num_clients}/" + f"local_rounds={lc_rds}"
    os.makedirs(exp_dir, exist_ok=True)

    print("======== evaluate =======")

    for policy_name in policies:
        policy, name = policy_name[0], policy_name[1]
        print("policy name:", name)
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
        with open(os.path.join(exp_dir, name + f"_{args.dataset}_result.json"), "w") as f:
            json.dump(result_dict, f, indent=4)

    from plot import save_figs
    save_figs('results/')
