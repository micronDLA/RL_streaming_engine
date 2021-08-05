## Streaming Engine Scheduler

- RL method that uses PPO to generate assignment for a given compute graph on a streaming engine topology.
- Compute graph can be changed in the ```train.py```
- Main hyperparameter to adjust is ```entropy_loss_factor``` and ```ppo_epoch```
- Modify ```calculate_reward()``` to change how the streaming engine behavior is represented in training.
- Optimal Transport and Sinkhorn Iterative Normalization code copied from SuperGlue's codebase.

NOTE: If you feel like the training is stuck and is not improving; wait for quiet a bit before you stop the training progress. Sometimes it will eventually figure out a good placement.

