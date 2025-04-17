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
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from tools import *
from env import *
from policy import *
from net import *

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
    state_dim = 40
    # === 指定 actor 以及对应optim ===
    actor, critic = choose_actor_critic(state_dim, action_dim, args)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化RL模型 ===
    policy = choose_policy(actor, actor_optim, critic, critic_optim, args)

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(512, 10), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    if not args.evaluate:
        if not args.no_logger:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            exp_dir = f"exp/{timestamp}"
            writer = SummaryWriter(exp_dir)
            logger = BasicLogger(writer)  # 创建基础日志记录器
        else:
            logger = None

    # PATH = os.path.join(exp_dir, 'atte' + "_ckpt.pth")
    # def save_best_fn(model):
    #     torch.save(model.state_dict(), PATH)
        
    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        repeat_per_collect=2,  # diff with offpolicy_trainer
        step_per_epoch=args.epochs,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=16,
        logger=logger,
        save_best_fn=None,
    )
    #


    # result = onpolicy_trainer(
    #     policy,
    #     train_collector,
    #     test_collector,
    #     max_epoch=50,
    #     step_per_epoch=10,  # 每个epoch进行多少次transitions
    #     # the number of repeat time for policy learning, for example, set it to 2 means the policy needs to learn each given batch data twice.
    #     repeat_per_collect=2,
    #     episode_per_test=1,  # 每次测试进行几个episode
    #     step_per_collect=20,  # 每次update前，收集多少step的数据
    #     batch_size=16,  # update的时候batch的大小
    #     logger=logger,
    #     save_best_fn=None,
    # )
    # exit()
    # for i in range(10):  # 训练总数
    #     train_collector.collect(n_step=1)
    #     print("===========")
    #     # 使用采样出的数据组进行策略训练
    #     losses = policy.update(1, train_collector.buffer,batch_size=1,repeat=1)
    #     exit()

    print("start!!!!!!!!!!!!!!!")
    for epoch, epoch_stat, info in trainer:
        print("=======\nEpoch:", epoch)
        print(epoch_stat)
        print(info)
        logger.write('epochs/stat', epoch, epoch_stat)
