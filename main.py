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
import argparse
from tianshou.policy import DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger

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

    # === 指定 actor 以及对应optim ===
    actor, critic = choose_actor_critic(state_dim, action_dim, args)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化RL模型 ===
    policy = choose_policy(actor, actor_optim, critic, critic_optim, args)

    # === 日志地址 模型存取 ===
    if args.evaluate or args.resume:
        exp_dir = args.ckpt_dir
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        exp_dir = f"exp/{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        with open(f"{exp_dir}/config.txt", "w") as f:
            config_des = f"dataset_{args.dataset}_epochs_{args.epochs}_algo_{args.algo}_dbranch_{args.dbranch}"
            f.write(f"{config_des}\n")
            f.write('args:\n')
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    PATH = os.path.join(exp_dir, args.algo + "_ckpt.pth")

    def save_best_fn(model):
        torch.save(model.state_dict(), PATH)

    if args.resume == True or args.evaluate == True:
        print("model path:", PATH)
        policy.load_state_dict(torch.load(PATH))
        print("model loaded!")

    # === tianshou 相关配置 ===
    buffer = VectorReplayBuffer(512, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer)
    test_collector = Collector(policy, test_envs, buffer=buffer)

    # === 训练 ===
    if not args.evaluate:
        if not args.no_logger:
            writer = SummaryWriter(exp_dir)
            logger = BasicLogger(writer)  # 创建基础日志记录器
        else:
            logger = None
        # result = offpolicy_trainer(
        #     policy,
        #     train_collector,
        #     test_collector,
        #     max_epoch=args.epochs,
        #     step_per_epoch=args.global_rounds,
        #     step_per_collect=args.global_rounds,
        #     episode_per_test=args.test_num,
        #     batch_size=args.datas_per_update,
        #     update_per_step=args.update_per_step,
        #     logger=logger,
        #     save_best_fn=save_best_fn,
        # )
        trainer = OffpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=args.epochs,
            step_per_epoch=args.global_rounds,
            step_per_collect=args.global_rounds,
            episode_per_test=args.test_num,
            batch_size=args.datas_per_update,
            update_per_step=args.update_per_step,
            logger=logger,
            save_best_fn=save_best_fn,
        )
        print("start!!!!!!!!!!!!!!!")
        for epoch, epoch_stat, info in trainer:
            print("=======\nEpoch:", epoch)
            print(epoch_stat)
            print(info)
            logger.write('epochs/stat', epoch, epoch_stat)

    # === eval ===
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
        act = policy(batch).act[0].cpu().detach().numpy()  # policy.forward 返回一个 batch，使用 ".act" 来取出里面action的数据
        obs, rew, done, done, info = env.step(act)
        for key, value in info.items():
            result_dict[key].append(value)
        if done:
            break
        env.close()

    print(result_dict)
    # res_dir = f"result/{timestamp}"
    with open(f"{exp_dir}/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
