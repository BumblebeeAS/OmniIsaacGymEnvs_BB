#!/bin/bash
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/mlpelu/nn/mlpelu.pth task.experiment_name='mlpelu' headless=True
# sleep 5
# python ../rlgames_train.py wandb_activate=True headless=True experiment='lstmelu64' train.params.network.mlp.activation='elu' train.params.network.rnn.units=64 && 
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstmelu64/nn/lstmelu64.pth task.experiment_name='lstmelu64' headless=True
# sleep 5
# python ../rlgames_train.py wandb_activate=True headless=True experiment='lstm'  && 
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstm/nn/lstm.pth task.experiment_name='lstm' headless=True 
# sleep 5
# python ../rlgames_train.py wandb_activate=True headless=True experiment='lstmelu' train.params.network.mlp.activation='elu' && 
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstmelu/nn/lstmelu.pth task.experiment_name='lstmelu' headless=True
# sleep 5
# python ../rlgames_train.py wandb_activate=True headless=True experiment='lstm64' train.params.network.rnn.units=64 && 
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstm64/nn/lstm64.pth task.experiment_name='lstm64' headless=True
# sleep 5 
# python ../rlgames_train.py wandb_activate=True headless=True experiment='lstmelu64' train.params.network.mlp.activation='elu' train.params.network.rnn.units=64 && 
# python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstmelu64/nn/lstmelu64.pth task.experiment_name='lstmelu64' headless=True

python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstmelu64/nn/lstmelu64.pth task.experiment_name='lstmelu64' headless=True train.params.network.mlp.activation='elu' train.params.network.rnn.units=64 
sleep 5
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstmelu/nn/lstmelu.pth task.experiment_name='lstmelu' headless=True train.params.network.mlp.activation='elu'
sleep 5
python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/lstm64/nn/lstm64.pth task.experiment_name='lstm64' headless=True train.params.network.rnn.units=64

python ../rlgames_train.py test=True num_envs=4 checkpoint=runs/mlpelu/nn/mlpelu.pth task.experiment_name='mlpelu' headless=True train.params.network.mlp.activation='elu'
sleep 5