# E. Culurciello
# April 2021

# Q-learn
# from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# TARGET_UPDATE = 10
RMSIZE = 10000 # replay memory size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNnet(nn.Module):

    def __init__(self, emb_size, in_size, out_size):
        super(DQNnet, self).__init__()
        self.l1 = nn.Linear(in_size, emb_size)
        self.l2 = nn.Linear(emb_size, emb_size)
        self.l3 = nn.Linear(emb_size, emb_size)
        self.lo = nn.Linear(emb_size, out_size)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.lo(x)


class DQNAgent(nn.Module):

    def __init__(self, args, env):
        super(DQNAgent, self).__init__()
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

        self.policy_net = DQNnet(args.emb_size, self.state_dim, self.action_dim).to(device)
        self.target_net = DQNnet(args.emb_size, self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.batch_size = 128#args.batch_size
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)#args.replay_memory_size)


    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



#state: [node] grid's flatten index
#action: where to put node
#rewards: cost
class Q_learn:
    def __init__(self, args, graph):
        self.args = args
        self.board = np.zeros(self.args.nodes)
        self.graph = graph
        self.states = []  # record hashes takes in an episode
        self.lr = args.q_lr
        self.exp_rate = args.exp_rate
        self.decay_gamma = args.decay_gamma
        self.states_value = {}  # state -> value
        self.reset()

    def reset(self):
        self.board[:] = -1
        self.states = []

    # get unique hash of state
    def get_hash(self, st):
        return str(st.flatten())

    #positions: array of avail idx
    #cur_state: [node] [idx]
    def chooseAction(self, positions, cur_state, node, exp=True):
        # choose action with most expected value
        action = 0
        if np.random.uniform(0, 1) <= self.exp_rate and exp: # exploration
            action = np.random.choice(positions)
        else: # exploitation
            value_max = -np.inf
            for p in positions:
                next_board = cur_state.copy()
                next_board[node] = p
                next_boardHash = self.get_hash(next_board)
                value = self.states_value.get(next_boardHash)
                if value is None:
                    value = 0
                # print("value", value)
                if value >= value_max: # do action with max value
                    value_max = value
                    action = p
            # print("{} takes action {}".format(self.name, action))
        return action

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

    # at the end, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    #get not used indexes
    def availablePositions(self):
        aa = [a for a in range(self.args.grid_depth * self.args.grid_size * self.args.grid_size)]
        for i in self.board:
            if i >= 0:
                aa.remove(i)
        return aa

    # get reward and update state_value table
    def giveReward(self):
        grid_in = []
        for i, idx in enumerate(self.board):
            c, y, x = np.unravel_index(int(idx), (self.args.grid_depth, self.args.grid_size, self.args.grid_size))
            grid_in.append([c, y, x])
        rwd = -calc_score(np.array(grid_in), self.graph, self.args)
        self.feedReward(rwd)
        return rwd

    def play(self, rounds=100):
        for i in range(rounds):
            for node in range(self.args.nodes):
                positions = self.availablePositions()
                p_action = self.chooseAction(positions, self.board, node)
                self.board[node] = p_action
                board_hash = self.get_hash(self.board)
                self.states.append(board_hash)

            rwd = self.giveReward()
            if i % 500 == 0:
                print(i, ' reward: ', rwd)
            self.reset()

    def test(self):
        self.reset()
        for node in range(self.args.nodes):
            positions = self.availablePositions()
            p_action = self.chooseAction(positions, self.board, node, exp=False)
            self.board[node] = p_action
        grid_in = []
        for i, idx in enumerate(self.board):
            c, y, x = np.unravel_index(int(idx), (self.args.grid_depth, self.args.grid_size, self.args.grid_size))
            grid_in.append([c, y, x])
        rwd = -calc_score(np.array(grid_in), self.graph, self.args)
        print('reward: ', rwd)

