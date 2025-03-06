# 定义随机分布的参数
import argparse


CPU_FREQUENCY_MEAN = 2e9  # Hz
CPU_FREQUENCY_STD = 0.5e9  # Hz
TRANSMIT_POWER_MEAN = 0.5  # W
TRANSMIT_POWER_STD = 0  # W
GAIN_MEAN = 5e-8
GAIN_STD = 1e-8
DISTANCE_MEAN = 5  # m
DISTANCE_STD = 2  # m

# 一个资源块的带宽
B = 1e6

# 总资源块数量
TOTAL_BLOCKS = 100


def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning Environment Simulation')

    # device(env AND train)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    # env
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                        help="Dataset to use.")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients.")
    parser.add_argument('--alpha', type=float, default=0.5, help="Dirichlet alpha for non-IID splitting.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader.NOT RL!!!")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory to store datasets.")
    parser.add_argument('--non_iid', action='store_true', default=True, help="Use non-IID split if set.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--log_fed', action='store_true', default=False, help="log federated process")
    parser.add_argument('--log_client', action='store_true', default=False, help="log client loss/T/E")
    parser.add_argument('--local_rounds', type=int, default=1, help='Number of local training rounds')
    parser.add_argument('--global_rounds', type=int, default=10, help='Number of global training rounds')
    parser.add_argument('--per_round_c_fraction', type=float, default=0.1,
                        help='Fraction of clients to select per round')
    parser.add_argument('--cycles_per_sample', type=int, default=1e5, help='cpu cycles for per sample')
    parser.add_argument('--fed_lr', type=float, default=0.01, help="federated learning rate")
    parser.add_argument('--fed_optim', type=str, default='adam',
                        choices=['adam', 'sgd'], help="Optimizer for federated learning")
    parser.add_argument('--rew_alpha',type=float, default=1., help="reward of acc")
    parser.add_argument('--rew_beta',type=float, default=1., help="reward of time")
    parser.add_argument('--rew_gamma',type=float, default=1., help="reward of energy")


    # rl traning
    parser.add_argument('--algo',type=str, default='diff_sac')
                        
    parser.add_argument('--epochs', type=int, default=3, help=':is epsilons')
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--step_per_collect', type=int, default=1, help='idk how to set')
    parser.add_argument('--training_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--test_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--datas_per_update', type=int, default=4, help='may large is ok.NOT DATALOADER!')
    parser.add_argument('--update_per_step', type=float, default=1, help='')
    
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')
    parser.add_argument('--ckpt_path', type=str, default='./diff_sacckpt.pth', help='save path')
    
    return parser.parse_args()
