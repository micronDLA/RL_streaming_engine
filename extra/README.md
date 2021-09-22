# Placement
place nodes in a PE grid to minimize data transfer hops among grid's PE

## Random search

`python3 grid_placement.py --mode 0`

## ES search

`python3 grid_placement.py --mode 1`

## RL PPO

`python3 grid_placement.py --mode 3`

## RL Q-learn

`python3 grid_placement.py --mode 4`

## run all methods

`python3 grid_placement.py --mode 2`

## Debug
See the graph and all prints

`python3 grid_placement.py --mode 2 --debug`
