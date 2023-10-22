#!/bin/bash

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp7' train.params.config.gamma=0.99 train.params.config.tau=0.95 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp7/nn/hp7.pth task.experiment_name='hp7' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp8' train.params.config.gamma=0.99 train.params.config.tau=0.9 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp8/nn/hp8.pth task.experiment_name='hp8' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp9' train.params.config.gamma=0.99 train.params.config.tau=0.99 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp9/nn/hp9.pth task.experiment_name='hp9' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp10' train.params.config.gamma=0.95 train.params.config.tau=0.95 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp10/nn/hp10.pth task.experiment_name='hp10' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp11' train.params.config.gamma=0.95 train.params.config.tau=0.9 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp11/nn/hp11.pth task.experiment_name='hp11' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='hp12' train.params.config.gamma=0.95 train.params.config.tau=0.99 && 
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/hp12/nn/hp12.pth task.experiment_name='hp12' headless=True
sleep 5

