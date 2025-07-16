import sys  # noqa
import os
from types import SimpleNamespace

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
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger

from env import *
from policy import *
from net import *

from datetime import datetime

def load_args_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    args_dict = {}
    in_args_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("args:"):
            in_args_section = True
            continue
        if in_args_section and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # 转换数据类型
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "none":
                value = None
            elif "." in value and value.replace(".", "", 1).isdigit():
                value = float(value)
            elif value.isdigit():
                value = int(value)

            args_dict[key] = value
            # 强制移除 ckpt_dir
    args_dict.pop("ckpt_dir", None)
    return SimpleNamespace(**args_dict)
            
if __name__ == "__main__":
    
    args = get_args()
    if args.resume :
        exp_dir = args.ckpt_dir
        config_path = os.path.join(exp_dir, "config.txt")
        args = load_args_from_txt(config_path)



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.task == 'acc':
        args.rew_b = args.rew_c = args.rew_d = 0.0

    # === 指定 actor 以及对应optim ===
    num_choose = args.num_choose
    
    if args.LSTM:
        state_dim = (args.input_dim + args.hidden_size) * args.num_clients
    else:
        state_dim = args.input_dim * args.num_clients
    print(state_dim, num_choose, args)
    actor, critic = choose_actor_critic(state_dim, num_choose, args)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # === 初始化RL模型 ===
    policy = choose_policy(actor, actor_optim, critic, critic_optim, args)

    # env
    env, train_envs, test_envs = gen_env(args)
    
    # === 日志地址 模型存取 ===
    if args.evaluate or args.resume:
        exp_dir = args.ckpt_dir
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        exp_dir = f"exp/{args.algo}/{args.dataset}/alpha={str(args.dir_alpha)}/lrs={args.local_rounds}/{timestamp}"
        if not args.no_logger:
            os.makedirs(exp_dir, exist_ok=True)
            with open(f"{exp_dir}/config.txt", "w") as f:
                config_des = f"dataset_{args.dataset}_epochs_{args.epochs}_algo_{args.algo}"
                f.write(f"{config_des}\n")
                f.write('args:\n')
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")

    BEST_PATH = os.path.join(exp_dir, args.algo + "_best.pth")
    CKPT_PATH = os.path.join(exp_dir, args.algo + "_ckpt.pth")

    def save_best_fn(model):
        torch.save(model.state_dict(), BEST_PATH)

    def save_checkpoint_fn(model):
        torch.save(model.state_dict(), CKPT_PATH)

    if args.evaluate:
        print("model path:", BEST_PATH)
        policy.load_state_dict(torch.load(BEST_PATH))
    if args.resume:
        print("model path:", CKPT_PATH)
        policy.load_state_dict(torch.load(CKPT_PATH))

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

        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=args.epochs,
            repeat_per_collect=2,  # diff with offpolicy_trainer
            step_per_epoch=args.global_rounds,
            episode_per_collect=1,
            episode_per_test=args.test_num,
            batch_size=args.rl_batch_size,
            logger=logger,
            save_best_fn=save_best_fn,
            # save_checkpoint_fn=save_checkpoint_fn,
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
    torch.manual_seed(args.seed)
    policy.load_state_dict(torch.load(BEST_PATH))
    policy.eval()
    env.is_fed_train = True
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
        if isinstance(act, torch.Tensor):
            act = act.cpu().detach().numpy()  # policy.forward 返回一个 batch，使用 ".act" 来取出里面action的数据
        obs, rew, done, done, info = env.step(act)
        for key, value in info.items():
            if key in result_dict:
                result_dict[key].append(value)
        if done:
            break
    env.close()

    print(result_dict)
    # res_dir = f"result/{timestamp}"
    tstmp = now.strftime("%m%d%H%M%S")
    with open(f"{exp_dir}/{args.algo}_{args.dataset}_result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
