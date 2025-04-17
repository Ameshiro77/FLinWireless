import numpy as np
import torch


def allocate_bandwidth(probs, total_bandwidth):
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    if all(type(x) is int for x in probs):
        return probs
    alloc = np.floor(probs * total_bandwidth).astype(np.int32)
    return alloc


def minmax_normalize(x: np.ndarray, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)


def zscore_normalize(x: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    return (x - mean) / std
