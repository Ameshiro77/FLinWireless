# 定义随机分布的参数
import argparse


CPU_FREQUENCY_MEAN = 2e9  # Hz
CPU_FREQUENCY_STD = 1e8  # Hz
TRANSMIT_POWER_MEAN = 1.5  # W
TRANSMIT_POWER_STD = 0.5  # W
GAIN_MAX = 1e-4
GAIN_MIN = 1e-6
DISTANCE_MEAN = 1000  # m
DISTANCE_STD = 0  # m
N0 = 4e-18  # -174dbm/HZ = 4e-21 W/HZ
CYCLES_PER_SAMPLE = 1e5
SEED = 42

# 一个资源块的带宽
B = 2e4

# 总资源块数量
TOTAL_BLOCKS = 1000


def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning Environment Simulation')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--task', type=str, default='hybrid', choices=['hybrid', 'acc'], help="Task type.")

    # device(env AND train)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    # env
    parser.add_argument('--base_noise', type=float, default=0.01, help="transmit noise")

    # fed
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'Fashion', 'CIFAR10'],
                        help="Dataset to use.")
    parser.add_argument('--num_clients', type=int, default=100, help="Number of clients.")
    parser.add_argument('--num_choose', type=int, default=10, help='Number of clients to choose per round')
    # parser.add_argument('--method', type=str, default='dirichlet', choices=['dirichlet', 'shards'],help="Partition method to use.")
    parser.add_argument('--dir_alpha', type=float, default=0.5, help="Dirichlet alpha for non-IID splitting.") #如果为0就是shards,num=2
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for DataLoader.NOT RL!!!")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory to store datasets.")
    parser.add_argument('--non_iid', action='store_true', default=True, help="Use non-IID split if set.")
    parser.add_argument('--log_fed', action='store_true', default=False, help="log federated process")
    parser.add_argument('--log_client', action='store_true', default=False, help="log client loss/T/E")
    parser.add_argument('--local_rounds', type=int, default=1, help='Number of local training rounds')
    parser.add_argument('--global_rounds', type=int, default=20, help='Number of global training rounds')
    parser.add_argument('--fed_lr', type=float, default=0.001, help="federated learning rate")
    parser.add_argument('--fed_optim', type=str, default='adam',
                        choices=['adam', 'sgd'], help="Optimizer for federated learning")
    parser.add_argument('--fed_train', action='store_true', default=False,
                        help="may need not train if state dont require 'loss' and rew dont require 'acc'!!")
    parser.add_argument('--add_noise', action='store_true', default=False, help="add noise in fed process")
    parser.add_argument('--hungarian', action='store_true', default=False, help="use hungarian algorithm")
    parser.add_argument('--update_gain', action='store_true', default=False, help="update gain")

    # reward
    parser.add_argument('--norm_cost', action='store_true', default=False, help="normalize cost")
    parser.add_argument('--acc_delta', action='store_true', default=False,
                        help="use △acc( U(acc_t+1)-U(acc_t) ) instead of a^acc")
    parser.add_argument('--return_coef', type=float, default=0.99, help='return coefficient')
    parser.add_argument('--rew_a', type=float, default=1., help="reward of quality")
    parser.add_argument('--rew_b', type=float, default=0.5, help="reward of time")
    parser.add_argument('--rew_c', type=float, default=0.5, help="reward of energy")
    parser.add_argument('--rew_d', type=float, default=0.2, help="reward of penalty")
    parser.add_argument('--penalty_coef', type=float, default=2., help="parti penalty")

    # args.select_input_dim, args.select_hidden_dim, args.select_num_heads, args.select_num_layers, args.num_selects
    # net
    parser.add_argument('--input_dim', type=int, default=5, help="client feature dim,final dim = i + h_lstm")
    parser.add_argument('--window_size', type=int, default=5, help="window size of history data")  # used for lstm
    parser.add_argument('--hidden_size', type=int, default=10, help="hidden size of lstm")

    # select
    parser.add_argument('--select_hidden_dim', type=int, default=128, help="hidden dim of select net")
    parser.add_argument('--select_num_heads', type=int, default=4, help="num_heads of select net")
    parser.add_argument('--select_num_layers', type=int, default=3, help="num_layers of select net")
    parser.add_argument('--select_norm_type', type=str, default='layer', choices=['layer', 'batch', 'rms'])
    parser.add_argument('--select_decoder', type=str, default='mask', choices=['mask', 'single'])

    # alloc
    parser.add_argument('--alloc_hidden_dim', type=int, default=128, help="hidden dim of select net")
    parser.add_argument('--alloc_num_heads', type=int, default=1, help="num_heads of select net")
    parser.add_argument('--alloc_num_layers', type=int, default=2, help="num_layers of select net")
    parser.add_argument('--alloc_norm_type', type=str, default='layer', choices=['layer', 'batch', 'rms'])

    # rl traning
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--alloc_steps', type=int, default=20)
    # parser.add_argument('--dbranch', action="store_true", default=False, help='if use 2 branch')
    # parser.add_argument('--threshold', type=float, default=0.8)
    # parser.add_argument('--lambda_1', type=float, default=1.0, help='loss align')
    # parser.add_argument('--lambda_2', type=float, default=1.0, help='loss boost')
    parser.add_argument('--no_logger', action='store_true', default=False, help='no tensorboard')

    parser.add_argument('--epochs', type=int, default=80, help=':is epsilons')
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--rl_batch_size', type=int, default=16, help='rl batch size')
    parser.add_argument('--step_per_collect', type=int, default=20, help='')
    parser.add_argument('--episode_per_collect', type=int, default=1, help='')  # 这两个只能取一个。

    parser.add_argument('--training_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--test_num', type=int, default=1, help='testing epochs')
    parser.add_argument('--datas_per_update', type=int, default=16, help='batch size')
    parser.add_argument('--update_per_step', type=float, default=1, help='')

    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='save path')
    parser.add_argument('--remark', type=str, default='', help='remark')

    return parser.parse_args()
