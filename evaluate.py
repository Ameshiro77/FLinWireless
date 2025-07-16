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
from tianshou.data import Collector
import re
from types import SimpleNamespace

AVG = False


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
    if 'LSTM' not in args_dict:
        args_dict['LSTM'] = True
    # 强制移除 ckpt_dir
    args_dict.pop("ckpt_dir", None)

    return SimpleNamespace(**args_dict)


if __name__ == "__main__":
    args = get_args()
    root_exp_dir = args.ckpt_dir  # 顶层目录
    print("Experiment root:", root_exp_dir)

    def is_leaf_dir(d):
        return all(not os.path.isdir(os.path.join(d, f)) for f in os.listdir(d))

    for subdir, _, files in os.walk(root_exp_dir):
        if not is_leaf_dir(subdir):
            continue
        if not any(f.endswith(".pth") for f in files):
            continue

        # === 加载 config.txt ===
        config_path1 = os.path.join(subdir, "config.txt")
        config_path2 = os.path.join(subdir, "data", "config.txt")
        if os.path.exists(config_path1):
            args = load_args_from_txt(config_path1)
        elif os.path.exists(config_path2):
            args = load_args_from_txt(config_path2)
        else:
            print(f"No config.txt found in {subdir}, skipping.")
            continue

        print("Loaded args from:", subdir)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.task == 'acc':
            args.rew_b = args.rew_c = args.rew_d = 0.0

        # 初始化模型
        num_choose = args.num_choose
        if args.LSTM:
            state_dim = (args.input_dim + args.hidden_size) * args.num_clients
        else:
            state_dim = args.input_dim * args.num_clients
        actor, critic = choose_actor_critic(state_dim, num_choose, args)
        actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        policy = choose_policy(actor, actor_optim, critic, critic_optim, args)

        # 初始化环境
        env, train_envs, test_envs = gen_env(args)

        # 遍历子目录下的所有 .pth 模型
        for f in files:
            if not f.endswith(".pth"):
                continue
            pth_path = os.path.join(subdir, f)
            print("Loading model:", pth_path)
            try:
                policy.load_state_dict(torch.load(pth_path, map_location="cpu"))
                print("Model loaded.")
            except Exception as e:
                print(f"Failed to load {pth_path}: {e}")
                continue

            # Evaluate
            print("======== evaluate =======")
            policy.eval()
            obs, info = env.reset()
            result_dict = {
                'current_round': [],
                'global_accuracy': [],
                'total_time': [],
                'total_energy': [],
                'reward': [],
            }
            rec_act = []
            done = False
            while not done:
                batch = Batch(obs=[obs], info=info)
                act = policy(batch).act[0]
                if isinstance(act, torch.Tensor):
                    act = act.cpu().detach().numpy()
                if AVG == True:
                    act[1] = np.full(num_choose, 1.0 / num_choose)
                rec_act.append(act[0])
                obs, rew, done, done, info = env.step(act)
                for key, value in info.items():
                    if key in result_dict:
                        result_dict[key].append(value)
                if done:
                    break
            env.close()
            del env, train_envs, test_envs
            del actor, critic, actor_optim, critic_optim, policy
            print("Eval done.")
            returns = []
            G = 0
            for rew in reversed(result_dict["reward"]):
                G = rew + 0.99 * G
                returns.insert(0, G)
            print("Return:", G)

            if AVG == False:
                with open(os.path.join(subdir, f.replace(".pth", "_result.json")), "w") as f_out:
                    json.dump(result_dict, f_out, indent=4)
            else:
                with open(os.path.join(subdir, f.replace(".pth", "_avg_result.json")), "w") as f_out:
                    json.dump(result_dict, f_out, indent=4)
            # with open(os.path.join(subdir, f.replace(".pth", "_recact.json")), "w") as f_out:
            #     json.dump(rec_act, f_out, indent=4)
