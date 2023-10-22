#!/bin/bash

python ../rlgames_train.py wandb_activate=True headless=True experiment='dr4' &&
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/dr4/nn/dr4.pth task.experiment_name='dr4' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='dr5' task.domain_random.scale=0.5 &&
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/dr5/nn/dr5.pth task.experiment_name='dr5' headless=True
sleep 5

python ../rlgames_train.py wandb_activate=True headless=True experiment='dr6' task.domain_random.curriculum=True &&
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/dr6/nn/dr6.pth task.experiment_name='dr6' headless=True
sleep 5


