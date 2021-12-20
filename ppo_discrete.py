# E. Culurciello
# A. Chang
# April 2021

# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch
# discrete version!

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch.distributions import Categorical # discrete
import numpy as np
from net import NormalHashLinear, TransformerModel
from dgl import nn as gnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.graphs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.graphs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class GraphEmb_Conv(nn.Module):
    def __init__(self, graph_emb, dropout=0.2):
        super(GraphEmb_Conv, self).__init__()
        self.cg_conv = nn.Conv1d(2, graph_emb, 1) #2: g.edges src node, dst node
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(graph_emb)

    def forward(self, x):
        x = self.cg_conv(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(self.norm(x))
        return x

class ACFF_SP(nn.Module): # feedforward ppo superposed model
    def __init__(self, in_dim, emb_size, out_dim, ntasks, mode='soft'):
        # ntasks: number of tasks
        super(ACFF_SP, self).__init__()
        self.fc = NormalHashLinear(in_dim, emb_size, ntasks)
        self.tanh = nn.Tanh()
        self.fc1 = NormalHashLinear(emb_size, emb_size, ntasks)
        self.fc2 = NormalHashLinear(emb_size, out_dim, ntasks)
        self.soft = nn.Softmax(dim=-1)  # if discrete
        self.emb_size = emb_size
        self.mode = mode

    def forward(self, x, task):
        y = self.tanh(self.fc(x, task))
        y = self.tanh(self.fc1(y, task))
        y = self.fc2(y, task)
        if self.mode == 'soft':
            y = self.soft(y)
        return y

class ACFF(nn.Module): # feedforward ppo
    def __init__(self, in_dim, emb_size, out_dim, mode='soft'):
        super(ACFF, self).__init__()
        self.fc = nn.Linear(in_dim, emb_size)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(emb_size, emb_size)
        self.fc2 = nn.Linear(emb_size, out_dim)
        self.soft = nn.Softmax(dim=-1)  # if discrete
        self.emb_size = emb_size
        self.mode = mode

    def forward(self, x):
        y = self.tanh(self.fc(x))
        y = self.tanh(self.fc1(y))
        y = self.fc2(y)
        if self.mode == 'soft':
            y = self.soft(y)
        return y

class ACRNN(nn.Module): # rnn ppo
    def __init__(self, in_dim, emb_size, out_dim, mode='soft'):
        super(ACRNN, self).__init__()

        # action mean range -1 to 1
        self.fc = nn.Linear(in_dim, emb_size)
        self.tanh = nn.Tanh()
        self.fc1 = nn.LSTM(emb_size, emb_size)
        self.fc2 = nn.Linear(emb_size, out_dim)
        self.soft = nn.Softmax(dim=-1)  # if discrete
        self.emb_size = emb_size
        self.mode = mode
        self.nlstm = 1
        self.hidden = (torch.zeros(self.nlstm, 1, emb_size), torch.zeros(self.nlstm, 1, emb_size))

    def reset_lstm(self):
        self.hidden = (torch.zeros(self.nlstm, 1, self.emb_size), torch.zeros(self.nlstm, 1, self.emb_size))

    def forward(self, x):
        y = self.tanh(self.fc(x))
        y, self.hidden = self.fc1(y, self.hidden)
        y = self.fc2(y)
        if self.mode == 'soft':
            y = self.soft(y)
        return y

class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, graph_size,
                 mode = 'linear', ntasks = 1):
        super(ActorCritic, self).__init__()
        self.device = device
        # self.graph_model = GraphEmb_Conv(graph_size) 
        self.graph_model = nn.ModuleList([
            gnn.SGConv(48, 64, 1, False, nn.ReLU),
            gnn.SGConv(64, 128, 1, False, nn.ReLU)
        ])
        self.graph_avg_pool = gnn.AvgPooling()
        if mode == 'rnn':
            self.actor = ACRNN(state_dim+graph_size, emb_size, action_dim, mode='soft')
            self.critic = ACRNN(state_dim + graph_size, emb_size, 1, mode='')

        elif mode == 'transformer':
            # ntokens: 1hot device topology
            self.model = TransformerModel(ntoken=action_dim, ninp=16, nhead=4, nhid=emb_size, nlayers=2)
            self.actor = ACFF(16*state_dim+graph_size, emb_size, action_dim, mode='soft')
            self.critic = ACFF(16*state_dim+graph_size, emb_size, 1, mode='')

        elif mode == 'super':
            self.actor = ACFF_SP(state_dim + graph_size, emb_size, action_dim, ntasks=ntasks, mode='soft')
            self.critic = ACFF_SP(state_dim + graph_size, emb_size, 1, ntasks=ntasks, mode='')
        else:
            self.actor = ACFF(state_dim+graph_size, emb_size, action_dim, mode='soft')
            self.critic = ACFF(state_dim+graph_size, emb_size, 1, mode='')
        self.mode = mode

    def reset_lstm(self):
        self.actor.reset_lstm()
        self.critic.reset_lstm()

    def forward(self):
        raise NotImplementedError

    def act_seq(self, state, graph_info):
        state_in = state[0].to(self.device)
        mask = state[1].to(self.device)
        emb = self.graph_model(graph_info).squeeze()
        o = self.model(state_in)
        o = o.view(-1)
        act_in = torch.cat((o, emb))
        action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate_seq(self, state, action, graph_info):
        state_in = state[0]
        mask = state[1]
        emb = self.graph_model(graph_info)
        o = self.model(state_in)
        o = torch.permute(o, (1, 0, 2)).contiguous()
        o = o.view(o.shape[0], -1)
        act_in = torch.cat((o[mask], emb), dim=1)
        action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(act_in)
        return action_logprobs, state_values, dist_entropy

    def act(self, state, graph_info, taskid=None):
        graph = dgl.add_self_loop(graph_info)
        graph_feat = graph.ndata['feat']
        for layer in self.graph_model:
            graph_feat = layer(graph, graph_feat)
        graph_feat = self.graph_avg_pool(graph, graph_feat)
        graph_feat = graph_feat.squeeze()
        # emb = self.graph_model(graph_info).squeeze()
        # print('[INFO] state shape', state.shape)
        # print('[INFO] graph_feat shape', graph_feat.shape)
        act_in = torch.cat((state, graph_feat))
        if self.mode == 'super':
            act_in = act_in.unsqueeze(0)
            action_probs = self.actor(act_in, taskid)
        else:
            action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, graph_info, taskid=None):
        graph = graph_info[0]
        graph = dgl.add_self_loop(graph)
        graph_feat = graph.ndata['feat']
        for layer in self.graph_model:
            graph_feat = layer(graph, graph_feat)
        graph_feat = self.graph_avg_pool(graph, graph_feat)
        # emb = self.graph_model(graph_info)
        act_in = torch.cat((state, graph_feat.broadcast_to(state.shape[0], -1)), dim=1)
        if self.mode == 'super':
            action_probs = self.actor(act_in, taskid)
        else:
            action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        if self.mode == 'super':
            state_values = self.critic(act_in, taskid)
        else:
            state_values = self.critic(act_in)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, args, state_dim, action_dim, mode='', ntasks = 1):
        #args.emb_size, betas, lr, gamma, K_epoch, eps_clip, loss_value_c, loss_entropy_c
        #ntasks: number of different graphs
        self.args = args
        self.device = device
        self.ntasks = ntasks
        self.state_dim = state_dim #input ready time (nodes, 1)
        self.action_dim = action_dim #output (nodes, 48)

        self.buffer = RolloutBuffer()
        self.ntokens = args.device_topology
        self.policy = ActorCritic(self.device,
                                  self.state_dim,
                                  self.args.emb_size,
                                  self.action_dim,
                                  self.args.graph_size,
                                  mode=mode,
                                  ntasks=ntasks).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device,
                                      self.state_dim,
                                      self.args.emb_size,
                                      self.action_dim,
                                      self.args.graph_size,
                                      mode=mode,
                                      ntasks=ntasks).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if args.model != '':
            self.load(args.model)

        self.MseLoss = nn.MSELoss()
        self.mode = mode

    def reset_lstm(self):
        if self.mode == 'rnn':
            self.policy.reset_lstm()
            self.policy_old.reset_lstm()

    def get_coord(self, assigment, action, node, grid_shape):
        # put node assigment to vector of node assigments
        action[node] = torch.tensor(np.unravel_index(assigment, grid_shape))
        return action

    def select_action(self, state, graph_info, taskid=None, mask=None):
        with torch.no_grad():
            graph_info = graph_info.to(self.device)
            if self.mode=='transformer':
                action, action_logprob = self.policy_old.act_seq(state, graph_info)
            else:
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state, graph_info, taskid)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.graphs.append(graph_info)
        self.buffer.logprobs.append(action_logprob)
        return action.item()
    
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # rewards = rewards.float().squeeze()
        
        # convert list to tensor
        old_masks = 0
        if self.mode == 'transformer':
            s = [i for i, _ in self.buffer.states]
            old_states = torch.squeeze(torch.stack(s, dim=0)).detach().to(self.device)
            old_states = torch.permute(old_states, (1, 0, 2))
            m = [i for _, i in self.buffer.states]
            old_masks = torch.squeeze(torch.stack(m, dim=0)).detach().to(self.device)
            old_graph = torch.squeeze(torch.stack(self.buffer.graphs, dim=0)).detach().to(self.device)
        else:
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
            old_graph = [graph.to(self.device) for graph in self.buffer.graphs]
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.args.K_epochs):

            # Evaluating old actions and values
            if self.mode == 'transformer':
                logprobs, state_values, dist_entropy = self.policy.evaluate_seq((old_states, old_masks), old_actions, old_graph)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_graph, taskid)

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
