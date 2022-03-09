# E. Culurciello
# A. Chang
# April 2021

# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch
# discrete version!

import dgl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch.distributions import Categorical # discrete
import numpy as np
from net import NormalHashLinear, TransformerModel
from dgl import nn as gnn
from util import ravel_index

_engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from modules import RolloutBuffer, ActorCritic

class PPO:
    def __init__(self,
                 args,
                 graphdef,
                 device,
                 state_dim,
                 mode='',
                 ntasks = 1):

        #ntasks: number of different graphs
        self.args = args
        self.graphdef = graphdef
        self.device_topology = device['topology']
        self.ntasks = ntasks
        self.state_dim = state_dim  # Input ready time (Number of tiles slices, 1)
        self.action_dim = device['action_dim'] #output (nodes, 48)
        self.gnn_in = graphdef['graph'].ndata['feat'].shape[1]
        self.buffer = RolloutBuffer()
        self.ntokens = args.device_topology

        self.policy = ActorCritic(args=args,
                                  device=device,
                                  state_dim=self.state_dim,
                                  emb_size=self.args.emb_size,
                                  action_dim=self.action_dim,
                                  graph_feat_size=self.args.graph_feat_size,
                                  gnn_in=self.gnn_in,
                                  mode=mode,
                                  ntasks=ntasks).to(_engine)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)

        self.policy_old = ActorCritic(args=args,
                                      device=device,
                                      state_dim=self.state_dim,
                                      emb_size=self.args.emb_size,
                                      action_dim=self.action_dim,
                                      graph_feat_size=self.args.graph_feat_size,
                                      gnn_in=self.gnn_in,
                                      mode=mode,
                                      ntasks=ntasks).to(_engine)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if args.model != '':
            self.load(args.model)

        self.MseLoss = nn.MSELoss()
        self.mode = mode

    def reset_lstm(self):
        if self.mode == 'rnn':
            self.policy.reset_lstm()
            self.policy_old.reset_lstm()

    """ Deprecated
    def get_coord(self, assigment, action, node):
        # put node assigment to vector of node assigments
        action[node] = torch.tensor(np.unravel_index(assigment, self.device_topology))
        return action
    """

    """ Old select_action
    def select_action(self, tensor_in, graphdef, node_id, action, pre_constr):
        with torch.no_grad():
            graph_info = graphdef['graph'].to(_engine)
            if self.mode=='transformer':
                action, action_logprob = self.policy_old.act_seq(tensor_in, graph_info)
            else:
                state = torch.FloatTensor(tensor_in).to(_engine)
                action, action_logprob = self.policy_old.act(state, graph_info, node_id, action, pre_constr)

        return action.item(), (state, action, graph_info, action_logprob)
    """

    def select_action(self, tensor_in, graphdef, node_id, mask):
        with torch.no_grad():
            graph_info = graphdef['graph'].to(_engine)
            if self.mode=='transformer':
                action, action_logprob = self.policy_old.act_seq(tensor_in, graph_info)
            else:
                state = torch.FloatTensor(tensor_in).to(_engine)
                mask = torch.tensor(mask, dtype=torch.bool).to(_engine)
                action, action_logprob = self.policy_old.act(state, graph_info, node_id, mask)

        return action.item(), (state, action, graph_info, action_logprob, mask)

    def add_buffer(self, inbuff, reward, done):
        state, action, graph_info, action_logprob, mask = inbuff
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.graphs.append(graph_info)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.masks.append(mask)


    def update(self, taskid=None):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(_engine)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # rewards = rewards.float().squeeze()

        # convert list to tensor
        old_masks = 0  # Used in transformer mode
        if self.mode == 'transformer':
            s = [i for i, _ in self.buffer.states]
            old_states = torch.squeeze(torch.stack(s, dim=0)).detach().to(_engine)
            old_states = torch.permute(old_states, (1, 0, 2))
            m = [i for _, i in self.buffer.states]
            old_masks = torch.squeeze(torch.stack(m, dim=0)).detach().to(_engine)
            old_graph = torch.squeeze(torch.stack(self.buffer.graphs, dim=0)).detach().to(_engine)
        else:
            old_masks = torch.squeeze(torch.stack(self.buffer.masks, dim=0)).detach().to(_engine)
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(_engine)
            old_graph = [graph.to(_engine) for graph in self.buffer.graphs]
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(_engine)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(_engine)

        # Optimize policy for K epochs
        for _ in range(self.args.K_epochs):

            # Evaluating old actions and values
            if self.mode == 'transformer':
                logprobs, state_values, dist_entropy = self.policy.evaluate_seq((old_states, old_masks), old_actions, old_graph)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_graph, old_masks, node_id=None, taskid=taskid)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                   + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                   - self.args.loss_entropy_c*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
