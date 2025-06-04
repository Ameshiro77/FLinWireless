#!/bin/bash
# if you use baselines algo,then ckpt should be None
#--hungarian \
# --ckpt_dir exp/ppo/CIFAR10/alpha=0.5/lrs=1 \
python evaluate.py \
    --evaluate \
    --ckpt_dir exp/ppo/Fashion \
   
    
