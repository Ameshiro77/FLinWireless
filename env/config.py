# 定义随机分布的参数
import argparse


CPU_FREQUENCY_MEAN = 2e9  # 2 GHz
CPU_FREQUENCY_STD = 0.5e9  # 0.5 GHz
TRANSMIT_POWER_MEAN = 5e-3  # 5 mW
TRANSMIT_POWER_STD = 2e-3  # 2 mW
GAIN_MEAN = 5e-8  # 5e-8
GAIN_STD = 1e-8  # 1e-8
DISTANCE_MEAN = 5  # 5 m
DISTANCE_STD = 2  # 2 m

# 一个资源块的带宽
B = 1e6

# 总资源块数量
TOTAL_BLOCKS = 100

def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning Environment Simulation')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                        help="Dataset to use.")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Dirichlet alpha for non-IID splitting.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory to store datasets.")
    parser.add_argument('--non_iid', action='store_true', default=True, help="Use non-IID split if set.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    parser.add_argument('--local_rounds', type=int, default=1, help='Number of local training rounds')
    parser.add_argument('--global_rounds', type=int, default=1, help='Number of global training rounds')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--per_round_c_fraction', type=float, default=0.1, help='Fraction of clients to select per round')
    parser.add_argument('--cycles_per_sample', type=int, default=1e5, help='cpu cycles for per sample')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    
    return parser.parse_args()