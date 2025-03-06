import random
from tianshou.policy import BasePolicy 
from env.config import *

class RandomPolicy(BasePolicy):
    def __init__(self, action_space ,args):
        super().__init__()
        self.num_clients = args.num_clients
        self.total_rb = TOTAL_BLOCKS
        self.action_space = action_space
    
    def learn(self):
        print("Learning...")

    def forward(self, obs, state=None, info={}, **kwargs):
        return self.action_space.sample(), state, info
    
    def random_action(self):
        action = [0] * self.num_clients
        indices = random.sample(range(self.num_clients), 5)
        for index in indices:
            action[index] = int(self.total_rb / self.num_clients)
        return action