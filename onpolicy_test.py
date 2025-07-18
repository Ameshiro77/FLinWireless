import torch
from tianshou.env import DummyVectorEnv
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.join(current_dir))  # noqa
sys.path.append(os.path.join(current_dir, "env"))  # noqa
sys.path.append(os.path.join(current_dir, "net"))  # noqa
sys.path.append(os.path.join(current_dir, "policy"))  # noqa
import gymnasium as gym
from gym import spaces
import torch.nn.functional as F
from env import *
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from policy import *
from tools import *
from net import *
from tools.misc import *
import json
from datetime import datetime
from SubAllocEnv import *
from net import *
from net.model import ObsProcessor


class AllocActor2(nn.Module):
    def __init__(
            self, state_dim, action_dim, hidden_dim=256, input_dim=14, embed_dim=64, window_size=5, hidden_size=10,
            norm_type='layer',
            is_LSTM=False):
        super().__init__()
        self.LSTM = is_LSTM
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder2 = SelectEncoder(input_dim=self.input_dim, embed_dim=embed_dim,
                                      num_heads=4, num_layers=2, norm_type=norm_type)

        self.decoder = DirichletPolicy(state_dim=embed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        # self.decoder = GaussianPolicy(state_dim=embed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.LSTMProcessor = ObsProcessor(window_size=window_size, hidden_size=hidden_size, lstm=self.LSTM)

    def forward(self, obs_dict, is_training=True):
        x = self.LSTMProcessor(obs_dict)
        x = self.embedding(x)  
        x = self.encoder2.forward(x)

        # x = self.embedding(x)
        # x, _ = self.encoder(x, x, x)

        # print(x.shape)
        # feature_dim = x.shape[-1]
        # x = x.view(x.shape[0], feature_dim, 10).permute(0, 2, 1)  # B * N * dim ->B * N * Input
        # x = self.encoder(x)  # -> B * N * H
        # x = x.mean(dim=1)  # -> B * H
        # x = x.reshape(x.shape[0], -1)  # B * N * dim ->B * (N * Input)

        action, logp, ent = self.decoder.forward(x, is_training)  # -> B * N * Action

        # print(action)
        return action, logp.unsqueeze(-1), ent


if __name__ == "__main__":

    # STATE 过高导致过拟合？

    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.num_clients = 10
    args.num_choose = 10
    args.algo = 'ppo'
    args.window_size = 5
    args.hidden_size = 5
    args.update_gain = True
    action_dim = 10

    env, train_envs, test_envs = make_sub_env(args)
    # exit()
    if args.LSTM:
        input_dim = 4 + args.hidden_size
    else:
        input_dim = 4
    state_dim = input_dim * args.num_clients
    # == actor
    from net.allocator import *
    # actor = DirichletPolicy(state_dim=state_dim, action_dim=args.num_choose, hidden_dim=256, constant=10).cuda()
    actor = AllocActor2(state_dim, 10, 256, input_dim, embed_dim=64,
                        window_size=args.window_size, hidden_size=args.hidden_size, is_LSTM=args.LSTM).cuda()
    # actor = MLPAllocator(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    # actor = DiffusionAllocor(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 1e-4)

    # === 指定 critic 以及对应optim ===
    critic = Critic_V(state_dim=state_dim, window_size=args.window_size,
                      hidden_size=args.hidden_size, LSTM=args.LSTM).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    policy = PPOPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        dist_fn=torch.distributions.Categorical,
        device=args.device,
        ent_coef=0.01,
        max_grad_norm=None,
        scheduler_iters=1000,
    )

    # )

    # baseline
    result_dict = {
        'total_time': [],
        'total_energy': [],
    }
    done = False
    rewards = []
    while not done:
        act = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        obs, rew, done, done, info = env.step(act)
        rewards.append(rew)
        for key, value in info.items():
            if key in result_dict:
                result_dict[key].append(value)
        if done:
            break
    env.close()
    returns = []
    G = 0
    for rew in reversed(rewards):
        G = rew + 0.99 * G
        returns.insert(0, G)  # 把新的G插到最前面
    print(G)
    with open(f"debug/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    buffer = VectorReplayBuffer(512, 1)
    train_collector = Collector(policy, train_envs, buffer=buffer)
    test_collector = Collector(policy, test_envs, buffer=buffer)
    # selects = random.sample(range(10), 5)
    print("=====")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    exp_dir = "debug/" + timestamp
    writer = SummaryWriter(exp_dir)
    logger = BasicLogger(writer)  # 创建基础日志记录器

    def save_best_fn(model):
        torch.save(model.state_dict(), exp_dir + "/best_model.pth")

    trainer = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        repeat_per_collect=2,  # diff with offpolicy_trainer
        step_per_epoch=100,
        episode_per_collect=1,
        episode_per_test=args.test_num,
        batch_size=32,
        logger=logger,
        save_best_fn=save_best_fn
    )

    print("start!!!!!!!!!!!!!!!")
    for epoch, epoch_stat, info in trainer:
        # selects = random.sample(range(10), 5)
        # train_envs.set_selects(selects)
        print("=======\nEpoch:", epoch)
        print(epoch_stat)
        print(info)
        logger.write('epochs/stat', epoch, epoch_stat)

    print("done")
    print(G)

    # === eval ===
    print("======== evaluate =======")
    np.random.seed(args.seed)
    policy.load_state_dict(torch.load(exp_dir + "/best_model.pth"))
    policy.eval()
    obs, info = env.reset()  # 重置环境，返回初始观测值
    result_dict = {
        'total_time': [],
        'total_energy': [],
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
    with open(f"{exp_dir}/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    # === PLOT

    import matplotlib.pyplot as plt

    # 加载两个 JSON 文件的结果
    with open(f"{exp_dir}/result.json", "r") as f:
        result_exp = json.load(f)
    with open("debug/result.json", "r") as f:
        result_debug = json.load(f)

    # 计算总时间与总能耗
    total_time_debug = sum(result_debug["total_time"])
    total_energy_debug = sum(result_debug["total_energy"])
    total_time_exp = sum(result_exp["total_time"])
    total_energy_exp = sum(result_exp["total_energy"])

    # 准备数据
    methods = ['Baseline', 'Trained']
    time_values = [total_time_debug, total_time_exp]
    energy_values = [total_energy_debug, total_energy_exp]

    # 画图
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = range(len(methods))
    width = 0.35

    # 左y轴：时间
    bar1 = ax1.bar([i - width/2 for i in x], time_values, width=width, color='blue', label='Total Time')
    ax1.set_ylabel("Total Time", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 右y轴：能耗
    ax2 = ax1.twinx()
    bar2 = ax2.bar([i + width/2 for i in x], energy_values, width=width, color='orange', label='Total Energy')
    ax2.set_ylabel("Total Energy", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # x轴设置
    plt.xticks(ticks=x, labels=methods)
    plt.title("Comparison of Total Time and Energy")
    fig.tight_layout()

    # 图例
    bars = bar1 + bar2
    labels = [bar.get_label() for bar in bars]
    plt.legend(bars, labels, loc="upper center")

    # 保存图像
    save_path = "debug/comparison.png"
    plt.savefig(save_path)
    plt.close()

    print(f"图像已保存至 {save_path}")

    # === ROUND-WISE PLOT ===
    # 每轮通信时间和能耗对比
    rounds = list(range(len(result_exp["total_time"])))
    plt.figure(figsize=(10, 5))

    # 通信时间
    plt.subplot(1, 2, 1)
    plt.plot(rounds, result_debug["total_time"], label="Baseline", marker='o', linestyle='--', color='blue')
    plt.plot(rounds, result_exp["total_time"], label="Trained", marker='s', linestyle='-', color='green')
    plt.xlabel("Round")
    plt.ylabel("Total Time")
    plt.title("Round-wise Total Time")
    plt.legend()
    plt.grid(True)

    # 通信能耗
    plt.subplot(1, 2, 2)
    plt.plot(rounds, result_debug["total_energy"], label="Baseline", marker='o', linestyle='--', color='orange')
    plt.plot(rounds, result_exp["total_energy"], label="Trained", marker='s', linestyle='-', color='red')
    plt.xlabel("Round")
    plt.ylabel("Total Energy")
    plt.title("Round-wise Total Energy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    line_save_path = "debug/comparison_roundwise.png"
    plt.savefig(line_save_path, dpi=400)
    plt.close()
    print(f"Round-wise comparison figure saved to {line_save_path}")
