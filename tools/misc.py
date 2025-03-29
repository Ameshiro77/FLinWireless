import numpy as np
import torch
def allocate_bandwidth(self, probs):
        """实现带宽分配算法：全分配，线性归一化分配带宽"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if all(type(x) is int for x in probs):
            return probs
        dbranch = self.args.dbranch
        if dbranch:
            alloc = np.floor(probs * self.total_bandwidth).astype(np.int32)
            return alloc
        # elif self.args.top_k == 0:
        #     n_clients = len(probs)
        #     allocations = []
        #     mask = (probs >= self.args.threshold).astype(
        #         float
        #     )  # 创建掩码，probs >= threshold 的位置为 1，否则为 0
        #     filtered_probs = probs * mask
        #     alloc = np.floor(filtered_probs * self.total_bandwidth).astype(int)
        #     return alloc
        else:
            alloc = np.floor(probs * self.total_bandwidth).astype(np.int32)
            return alloc