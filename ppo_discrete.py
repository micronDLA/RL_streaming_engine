# E. Culurciello
# A. Chang
# April 2021

# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch
# discrete version!

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch.distributions import Categorical # discrete
import gym
import numpy as np

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

class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, graph_size):
        self.device = device
        super(ActorCritic, self).__init__()
        self.graph_model = GraphEmb_Conv(graph_size)

        # action mean range -1 to 1
        self.actor = nn.Sequential(
                nn.Linear(state_dim+graph_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, action_dim),
                # nn.Tanh() # if continuous
                nn.Softmax(dim=-1) # if discrete
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim+graph_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, 1)
                )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, graph_info):
        emb = self.graph_model(graph_info).squeeze()
        act_in = torch.cat((state, emb))
        action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action, graph_info):
        emb = self.graph_model(graph_info)
        act_in = torch.cat((state, emb), dim=1)
        action_probs = self.actor(act_in)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(act_in)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, args, env):
        #args.emb_size, betas, lr, gamma, K_epoch, eps_clip, loss_value_c, loss_entropy_c
        self.args = args
        self.device = device

        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(self.device,
                                  self.state_dim,
                                  self.args.emb_size,
                                  self.action_dim,
                                  self.args.graph_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device,
                                      self.state_dim,
                                      self.args.emb_size,
                                      self.action_dim,
                                      self.args.graph_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if args.model != '':
            self.load(args.model)

        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, graph_info):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            graph_info = graph_info.to(self.device)
            action, action_logprob = self.policy_old.act(state, graph_info)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.graphs.append(graph_info)
        self.buffer.logprobs.append(action_logprob)

        return action.item()
    
    def update(self):
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
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_graph = torch.squeeze(torch.stack(self.buffer.graphs, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.args.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_graph)

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