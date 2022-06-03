# Streaming Engine Scheduler

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6611892.svg)](https://doi.org/10.5281/zenodo.6611892)


Gym environment for Streaming Engine and PPO code to generate assignment for a given compute graph on a streaming engine topology.

## Dependencies

First, install PyGraphViz on your system: https://pygraphviz.github.io/documentation/stable/install.html

Install require Python packages
```
pip install -r requirements.txt
```

## Experiments
Run training:
```
python train.py
```

See experiements results using:
```
tensorboard --logdir runs/ --bind_all
```
## Usage
```
usage: train.py [-h] [--device-topology DEVICE_TOPOLOGY [DEVICE_TOPOLOGY ...]] [--pipeline-depth PIPELINE_DEPTH]
                [--epochs EPOCHS] [--nodes NODES] [--debug] [--input INPUT] [--pass-timing] [--no-tm-constr]
                [--no-sf-constr] [--no-device-cross-connections] [--graph_feat_size GRAPH_FEAT_SIZE]
                [--emb_size EMB_SIZE] [--update_timestep UPDATE_TIMESTEP] [--K_epochs K_EPOCHS]
                [--eps_clip EPS_CLIP] [--gamma GAMMA] [--lr LR] [--betas BETAS] [--loss_entropy_c LOSS_ENTROPY_C]
                [--loss_value_c LOSS_VALUE_C] [--model MODEL] [--log_interval LOG_INTERVAL]

Streaming Engine RL Mapper

optional arguments:
  -h, --help            show this help message and exit
  --device-topology DEVICE_TOPOLOGY [DEVICE_TOPOLOGY ...]
                        Device topology of Streaming Engine
  --pipeline-depth PIPELINE_DEPTH
                        processing pipeline depth
  --epochs EPOCHS       number of epochs
  --nodes NODES         number of nodes
  --debug               enable debug mode
  --input INPUT         load input json from file
  --pass-timing         enable pass through timing
  --no-tm-constr        disable tile memory constraint
  --no-sf-constr        disable sync flow constraint
  --no-device-cross-connections
                        disable sync flow constraint
  --graph_feat_size GRAPH_FEAT_SIZE
                        graph_feat_size
  --emb_size EMB_SIZE   embedding size
  --update_timestep UPDATE_TIMESTEP
                        update policy every n episodes
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
