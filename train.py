import torch
import argparse
import numpy as np

from collections import deque
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from net import PolicyNet
from env import StreamingEngineEnv

#torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument('--device-topology', type=int, nargs=3, default=(8, 8, 3))
parser.add_argument('--device-cross-connections', action='store_true')

parser.add_argument('--num-episode', type=int, default=1000000)
parser.add_argument('--ppo-epoch', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip-eps', type=float, default=0.2)
parser.add_argument('--entropy-loss-factor', type=float, default=1e-2)
parser.add_argument('--max-grad-norm', type=float, default=1)

args = parser.parse_args()

# define the distance function graph
# src_ids = [0, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8,  8, 11, 12, 12, 13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 19, 20, 20, 21, 22]
# dst_ids = [1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 7, 8, 9, 10, 12, 13, 19, 15, 14, 16, 17, 18, 19, 17, 18, 19, 20, 21, 21, 22, 22, 23]
# sar_fn_graphdef = (src_ids, dst_ids)


# define the FFT graph
src_ids = [0, 1,
           3, 4, 5,
           7, 8, 8, 9, 10, 11, 12, 13, 14,
           16, 17, 18, 19, 20, 21, 22, 23, 19, 19,
           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
           37, 38, 38, 39, 40, 41, 41, 42, 42, 43, 45, 44, 46, 47, 48, 49, 50, 47, 48, 49, 50, 46, 51, 56, 57, 58,
           60, 61,
           63,
           65, 66,
          ]
dst_ids = [1, 2,
           4, 5, 6,
           8, 9, 10, 11, 11, 12, 13, 14, 15,
           17, 18, 19, 20, 21, 22, 23, 24, 21, 23,
           26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
           38, 39, 40, 41, 42, 43, 45, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 53, 52, 55, 54, 51, 56, 57, 58, 59,
           61, 62,
           64,
           66, 67,
          ]
sar_fn_graphdef = (src_ids, dst_ids)


# initialize Environment, Network and Optimizer
env    = StreamingEngineEnv(sar_fn_graphdef,
                            args.device_topology, args.device_cross_connections,
                            device_feat_size=48, graph_feat_size=32)
policy = PolicyNet(32, 64, 1, 48, 4, 128, 0.1, 4, 100)
optim  = Adam(policy.parameters(), lr=args.lr)

# to keep track of average reward
reward_buf = deque(maxlen=100)
reward_buf.append(0)

# train
for episode in range(args.num_episode):

    # reset env
    state = env.reset()

    # collect 'trajectory'
    # only one trajectory step
    with torch.no_grad():
        old_action, old_logp, old_entropy, old_scores = policy(*state)
        old_reward = env.step(old_action)

    # use 'trajectory' to train network
    for epoch in range(args.ppo_epoch):

        action, logp, entropy, scores = policy(*state)
        reward = env.step(action)

        ratio = torch.exp(logp) / (torch.exp(old_logp) + 1e-8)
        surr1 = ratio * reward
        surr2 = torch.clamp(ratio, 1-args.clip_eps, 1+args.clip_eps) * reward
        action_loss = -torch.fmin(surr1, surr2)
        entropy_loss = entropy * args.entropy_loss_factor

        loss = (action_loss + entropy)
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optim.step()

        reward_buf.append(reward.mean())

        print(episode, epoch, reward, loss.item(), np.mean(reward_buf))

    if episode % 10 == 0:
        plt.imshow(old_scores[0].exp().detach(), vmin=0, vmax=1)
        plt.pause(1e-6)
        #plt.show()

        #env.render()
