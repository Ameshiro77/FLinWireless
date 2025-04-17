#!/bin/bash

python main.py \
    --epochs 100 \
    --rew_a 1.3 \
    --rew_b 0.5 \
    --rew_c 0.5 \
    --rew_d 0.4 \
    --algo ppo \
    --constant 10