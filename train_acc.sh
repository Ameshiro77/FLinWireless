#!/bin/bash
#--fed_train \
#--add_noise \
#--hungarian \
# layer rms batch
# --acc_delta
#MNIST Fashion CIFAR10

python main.py \
    --epochs 70 \
    --task acc \
    --dataset CIFAR10 \
    --algo ppo \
    --fed_train \
    --dir_alpha 0.5 \
    --local_rounds 1 \
    --global_rounds 100 \
    --num_clients 100 \
    --num_choose 10 \
    --update_gain \
    --acc_delta \
    --rew_a 1.0 \
    --rew_b 0.0 \
    --rew_c 0.0 \
    --rew_d 0.0 \
    --actor_lr 1e-4 \
    --critic_lr 1e-3 \
    --input_dim 6 \
    --select_hidden_dim 128 \
    --select_num_heads 8 \
    --select_num_layers 3 \
    --select_norm_type layer \
    --select_decoder mask \
    --window_size 5 \
    --hidden_size 5 \
    --alloc_hidden_dim 64 \
    --alloc_num_heads 4 \
    --alloc_num_layers 2 \
    --alloc_norm_type layer \
    --episode_per_collect 1 \
    --rl_batch_size 32 \
    --remark 'jianglicost100->10'