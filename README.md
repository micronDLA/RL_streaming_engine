# Streaming Engine Scheduler

- Methods that use Random Search, ES and PPO to generate assignment for a given compute graph on a streaming engine topology.
- Main hyperparameter to adjust is ```entropy_loss_factor``` and ```ppo_epoch```
- Modify ```calculate_reward()``` to change how the streaming engine behavior is represented in training.
- Optimal Transport and Sinkhorn Iterative Normalization code copied from SuperGlue's codebase.

## Usage
```
usage: train.py [-h] [--mode MODE] [--device_topology DEVICE_TOPOLOGY [DEVICE_TOPOLOGY ...]] [--epochs EPOCHS] [--nodes NODES] [--debug] [--input INPUT] [--no-tm-constr] [--no-sf-constr] [--ppo-epoch PPO_EPOCH]
                [--max-grad-norm MAX_GRAD_NORM] [--graph_size GRAPH_SIZE] [--emb_size EMB_SIZE] [--update_timestep UPDATE_TIMESTEP] [--K_epochs K_EPOCHS] [--eps_clip EPS_CLIP] [--gamma GAMMA] [--lr LR] [--betas BETAS]
                [--loss_entropy_c LOSS_ENTROPY_C] [--loss_value_c LOSS_VALUE_C] [--model MODEL] [--log_interval LOG_INTERVAL]

Streaming Engine RL Mapper

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           0 - random search, 1 - CMA-ES search, 2 - RL PPO, 3 - sinkhorn, 4 - multigraph, 5 - transformer
  --device_topology DEVICE_TOPOLOGY [DEVICE_TOPOLOGY ...]
                        Device topology of Streaming Engine
  --epochs EPOCHS       number of epochs
  --nodes NODES         number of nodes
  --debug               enable debug mode
  --input INPUT         load input json from file
  --no-tm-constr        disable tile memory constraint
  --no-sf-constr        disable sync flow constraint
  --ppo-epoch PPO_EPOCH
  --max-grad-norm MAX_GRAD_NORM
  --graph_size GRAPH_SIZE
                        graph embedding size
  --emb_size EMB_SIZE   embedding size
  --update_timestep UPDATE_TIMESTEP
                        update policy every n timesteps
  --K_epochs K_EPOCHS   update policy for K epochs
  --eps_clip EPS_CLIP   clip parameter for PPO
  --gamma GAMMA         discount factor
  --lr LR               parameters for Adam optimizer
  --betas BETAS
  --loss_entropy_c LOSS_ENTROPY_C
                        coefficient for entropy term in loss
  --loss_value_c LOSS_VALUE_C
                        coefficient for value term in loss
  --model MODEL         load saved model from file
  --log_interval LOG_INTERVAL
                        interval for logging data
```

## Dependencies

First, install PyGraphViz on your system: https://pygraphviz.github.io/documentation/stable/install.html

Install require Python packages
```
pip install -r requirements.txt
```

## Experiments
See experiements results using:
```
tensorboard --logdir runs/ --bind_all
```
