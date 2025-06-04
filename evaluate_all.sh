#!/bin/bash

COMMON_ARGS="\
    --num_choose 10 \
    --num_clients 100 \
    --global_rounds 100 \
    --local_rounds 1 \
    --dir_alpha 0"

parallel -j 3 --lb --halt soon,fail=1 \
  'python eval_baselines.py '"$COMMON_ARGS"' --dataset {1}' ::: MNIST Fashion CIFAR10
