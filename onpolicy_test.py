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

from datetime import datetime
from SubAllocEnv import *
from net import *
from net.model import ObsProcessor


class AllocActor2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, input_dim=15, embed_dim=64, norm_type='layer'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder2 = SelectEncoder(input_dim=self.input_dim, embed_dim=embed_dim,
                                      num_heads=4, num_layers=2, norm_type=norm_type)

        self.decoder = DirichletPolicy(state_dim=embed_dim, action_dim=action_dim, hidden_dim=hidden_dim, constant=1)
        # self.decoder = GaussianPolicy(state_dim=embed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.LSTMProcessor = ObsProcessor(window_size=5, hidden_size=10)

    def forward(self, obs_dict, is_training=True):
        x = self.LSTMProcessor(obs_dict)

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
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.num_clients = 10
    args.num_choose = 10
    args.algo = 'ppo'
    args.update_gain = True
    action_dim = 10

    env, train_envs, test_envs = make_sub_env(args)
    # exit()
    input_dim = 14
    state_dim = input_dim * args.num_clients
    # == actor
    from net.allocator import *
    # actor = DirichletPolicy(state_dim=state_dim, action_dim=args.num_choose, hidden_dim=256, constant=10).cuda()
    actor = AllocActor2(state_dim, 10, 256, input_dim, embed_dim=64).cuda()
    # actor = MLPAllocator(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    # actor = DiffusionAllocor(state_dim=state_dim, action_dim=5, hidden_dim=256).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 1e-4)

    # === 指定 critic 以及对应optim ===
    critic = Critic_V(state_dim=state_dim,).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    policy = PPOPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        dist_fn=torch.distributions.Categorical,
        device=args.device,
        ent_coef=0.01,
        max_grad_norm=0.2,
        scheduler_iters=100,
    )

    # )

    done = False
    rewards = []
    while not done:
        act = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        obs, rew, done, done, info = env.step(act)
        rewards.append(rew)
        if done:
            break
    env.close()
    returns = []
    G = 0
    for rew in reversed(rewards):
        G = rew + 0.99 * G
        returns.insert(0, G)  # 把新的G插到最前面
    print(G)
    # exit()

    if args.evaluate:
        print("======== evaluate =======")
        policy.load_state_dict(torch.load("./debug/best_model.pth"))
        np.random.seed(args.seed)
        policy.eval()
        env.is_fed_train = True
        obs, info = env.reset()  # 重置环境，返回初始观测值
        result_dict = {
            "communication_rounds": [],
            "total_time": [],
            "total_energy": [],
            "sub_reward": []
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
        returns = []
        G = 0
        for rew in reversed(result_dict["sub_reward"]):
            G = rew + 0.99 * G
            returns.insert(0, G)  # 把新的G插到最前面
        print(G)

    else:
        buffer = VectorReplayBuffer(512, 1)
        train_collector = Collector(policy, train_envs, buffer=buffer)
        test_collector = Collector(policy, test_envs, buffer=buffer)
        # selects = random.sample(range(10), 5)
        print("=====")

        writer = SummaryWriter("./debug")
        logger = BasicLogger(writer)  # 创建基础日志记录器

        def save_best_fn(model):
            torch.save(model.state_dict(), "./debug/best_model.pth")

        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=args.epochs,
            repeat_per_collect=2,  # diff with offpolicy_trainer
            step_per_epoch=80,
            episode_per_collect=2,
            episode_per_test=args.test_num,
            batch_size=16,
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
