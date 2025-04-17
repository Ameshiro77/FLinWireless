# 定义随机分布的参数
import argparse


CPU_FREQUENCY_MEAN = 2e9  # Hz
CPU_FREQUENCY_STD = 0.5e9  # Hz
TRANSMIT_POWER_MEAN = 0.5  # W
TRANSMIT_POWER_STD = 0.1  # W
GAIN_MEAN = 5e-8
GAIN_STD = 1e-8
DISTANCE_MEAN = 150  # m
DISTANCE_STD = 20  # m

# 一个资源块的带宽
B = 1e6

# 总资源块数量
TOTAL_BLOCKS = 100


def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning Environment Simulation')

    # device(env AND train)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    # env
    parser.add_argument('--base_noise', type=float, default=0.01, help="transmit noise")

    # fed
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'],
                        help="Dataset to use.")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients.")
    parser.add_argument('--num_choose', type=int, default=5, help='Number of clients to choose per round')
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
    parser.add_argument('--fed_train', action='store_true', default=False,
                        help="may need not train if state dont require 'loss' and rew dont require 'acc'!!")

    # reward
    parser.add_argument('--return_coef', type=float, default=0.999, help='return coefficient')
    parser.add_argument('--rew_a', type=float, default=1., help="reward of quality")
    parser.add_argument('--rew_b', type=float, default=1., help="reward of time")
    parser.add_argument('--rew_c', type=float, default=1., help="reward of energy")
    parser.add_argument('--rew_d', type=float, default=1., help="reward of penalty")
    parser.add_argument('--penalty_coef', type=float, default=2., help="parti penalty")

    # net
    parser.add_argument('--constant', type=int, default=10, help="hidden dim of net")

    # task_type
    parser.add_argument('--no_allocation', action="store_true", default=False, help='without bandwidth allocation')
    parser.add_argument('--top_k', type=int, default=5, help='if set 0,no fixed topk.')

    # rl traning
    parser.add_argument('--algo', type=str, default='diff_sac')
    parser.add_argument('--alloc_steps', type=int, default=20)
    parser.add_argument('--dbranch', action="store_true", default=False, help='if use 2 branch')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--lambda_1', type=float, default=1.0, help='loss align')
    parser.add_argument('--lambda_2', type=float, default=1.0, help='loss boost')
    parser.add_argument('--no_logger', action='store_true', default=False, help='no tensorboard')

    parser.add_argument('--epochs', type=int, default=80, help=':is epsilons')
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--step_per_collect', type=int, default=20, help='idk how to set')
    parser.add_argument('--training_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--test_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--datas_per_update', type=int, default=16, help='batch size')
    parser.add_argument('--update_per_step', type=float, default=1, help='')

    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='save path')
    parser.add_argument('--remark', type=str, default='', help='remark')

    return parser.parse_args()
