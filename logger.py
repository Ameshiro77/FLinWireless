# 用于记录episode的reward的详细信息
import time
import math
from tianshou.utils import BasicLogger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class InfoLogger(BasicLogger):
    def __init__(self, writer: SummaryWriter):
        super().__init__(writer)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.log_file = f"./logs/exp/log_{timestamp}.txt"
        self.episode_rewards = []  # 存储每个 episode 的 reward
        self.episode_details = []  # 存储每个 episode 的细节（如模型精度、通信开销）

    def log_episode(self, episode_id, reward, info):
        self.episode_rewards.append(reward)
        self.episode_details.append(info)
        # self.write("train/reward", reward, episode_id)
        # self.write("train/accuracy", info["accuracy"], episode_id)
        # self.write("train/communication_cost", info["communication_cost"], episode_id)
        print(f"Episode {episode_id}: Reward={reward}, global accuracy={info['global_acc']}, global loss={info['global_loss']}\
           		 total_time={info['total_time']},total_energy={info['total_energy']}")

    # def write(self, key, value, step):
    #     # write to tensorboard;
    #     if self.writer is not None:
    #         self.writer.add_scalar(key, value, step)
    #     # write to file;
    #     self.writer.add_scalar(key, value, step)