import torch
import argparse
import numpy as np

from collections import deque
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from net import PolicyNet
from env import StreamingEngineEnv

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument('--device-topology', type=int, nargs=3, default=(4, 4, 3))

parser.add_argument('--num-episode', type=int, default=1000000)
parser.add_argument('--ppo-epoch', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip-eps', type=float, default=0.2)
parser.add_argument('--entropy-loss-factor', type=float, default=1e-2)
parser.add_argument('--max-grad-norm', type=float, default=1)

args = parser.parse_args()

# define the distance function graph
src_ids = [0, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8,  8, 11, 12, 13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 19, 20, 20, 21, 22]
dst_ids = [1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 7, 8, 9, 10, 12, 13, 15, 14, 16, 17, 18, 19, 17, 18, 19, 20, 21, 21, 22, 22, 23]
dist_fn_graphdef = (src_ids, dst_ids)

# initialize Environment, Network and Optimizer
env    = StreamingEngineEnv(args.device_topology, 48, dist_fn_graphdef, 32)
policy = PolicyNet(32, 64, 1, 48, 4, 128, 0.1, 4, 100)
optim  = Adam(policy.parameters(), lr=args.lr)

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
