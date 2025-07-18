#!/bin/bash
#acc_delta  dataset
#hybrid  alpha  rew  method algo
# ent_coef norm_xi LSTM

COMMON_ARGS="\
    --epochs 100 \
    --task acc \
    --fed_train \
    --algo ppo \
    --local_rounds 1 \
    --global_rounds 100 \
    --num_clients 100 \
    --num_choose 10 \
    --dataset CIFAR10 \
    --ent_coef 0.01 \
    --acc_delta \
    --norm_xi \
    --LSTM \
    --update_gain \
    --rew_a 1.0 \
    --rew_b 0.5 \
    --rew_c 0.5 \
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
    --alloc_method d \
    --alloc_hidden_dim 64 \
    --alloc_num_heads 4 \
    --alloc_num_layers 2 \
    --alloc_norm_type layer \
    --episode_per_collect 1 \
    --rl_batch_size 16"

parallel -j 3 --lb --halt soon,fail=1 \
    'python main.py '"$COMMON_ARGS"' --dir_alpha {1}  --remark normxi!' \
    ::: 0.0 0.5 1.0