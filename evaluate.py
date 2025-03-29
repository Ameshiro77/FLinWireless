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


if __name__ == "__main__":

    args = get_args()
    env, train_envs, test_envs = gen_env(args)
    # === 计算 state_dim action_dim ===
    state_dim = 0
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            state_dim += 1  # Discrete 空间是标量
        elif isinstance(space, spaces.Box):
            state_dim += np.prod(space.shape)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    action_dim = args.num_clients

    # # === 指定 actor 以及对应optim ===
    # actor, critic = choose_actor_critic(state_dim, action_dim, args)
    # actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # # === 初始化RL模型 ===
    # policy = choose_policy(actor, actor_optim, critic, critic_optim, args)
    policy = RandomPolicy(args).cuda()
    
    print("======== evaluate =======")
    np.random.seed(args.seed)
    policy.eval()
    obs, info = env.reset()  # 重置环境，返回初始观测值
    result_dict = {
        'current_round': [],
        'global_loss': [],
        'global_accuracy': [],
        'total_time': [],
        'total_energy': [],
        'env_all_time': [],
        'env_all_energy': [],
    }

    from tianshou.data import Batch
    import json
    # from tianshou.policy
    done = False
    while not done:
        batch = Batch(obs=[obs])  # 第一维是 batch size
        act = policy(batch).act[0]
        if isinstance(act,torch.Tensor):
            act = act. cpu().detach().numpy()  # policy.forward 返回一个 batch，使用 ".act" 来取出里面action的数据
        obs, rew, done, done, info = env.step(act)
        for key, value in info.items():
            result_dict[key].append(value)
        if done:
            break
    env.close()

    print(result_dict)
    # res_dir = f"result/{timestamp}"
    with open(f"baselines/rand_result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
