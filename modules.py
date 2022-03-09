import dgl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch import einsum
from torch.distributions import Categorical # discrete
from net import NormalHashLinear, TransformerModel
import numpy as np
from dgl import nn as gnn
from util import ravel_index
from einops import reduce

from typing import Optional

_engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.graphs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.graphs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]

class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)


class GraphEmb_Conv(nn.Module):
    def __init__(self, graph_emb, dropout=0.2):
        super(GraphEmb_Conv, self).__init__()
        self.cg_conv = nn.Conv1d(2, graph_emb, kernel_size=(1,1)) #2: g.edges src node, dst node
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(graph_emb)

    def forward(self, x):
        x = self.cg_conv(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(self.norm(x))
        return x


class PAM_ModuleM(nn.Module):
    def __init__(self, in_dim):
        super(PAM_ModuleM, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 5, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 5, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        out = torch.bmm(proj_value, attention)
        out = self.gamma * out + x
        out = out.permute(0, 2, 1)
        return out



class CAM_ModuleM(nn.Module):
    def __init__(self, in_dim):
        super(CAM_ModuleM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        proj_query = x.permute(0, 2, 1)
        proj_key = x
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x
        out = torch.bmm(proj_value, attention)
        out = self.gamma*out + x
        return out

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
    def __init__(self,
                 args,
                 device,
                 state_dim,
                 emb_size,
                 action_dim,
                 graph_feat_size,
                 gnn_in,
                 mode = 'linear',
                 ntasks = 1):
        super(ActorCritic, self).__init__()
        self.args = args
        self.device = device
        # self.graph_model = GraphEmb_Conv(graph_feat_size)
        self.graph_model = nn.ModuleList([
            gnn.SGConv(gnn_in, 64, 1, False, nn.ReLU),
            gnn.SGConv(64, 128, 1, False, nn.ReLU)
        ])
        act_feat_sz = graph_feat_size + 2  # graph feature + state: [readytime, node sel]
        self.pam_attention = PAM_ModuleM(act_feat_sz)
        self.cam_attention = CAM_ModuleM(act_feat_sz)

        self.graph_avg_pool = gnn.AvgPooling()
        if mode == 'rnn':
            self.actor = ACRNN(state_dim+graph_feat_size, emb_size, action_dim, mode='soft')
            self.critic = ACRNN(state_dim + graph_feat_size, emb_size, 1, mode='')

        elif mode == 'transformer':
            # ntokens: 1hot device topology
            self.model = TransformerModel(ntoken=action_dim, ninp=16, nhead=4, nhid=emb_size, nlayers=2)
            self.actor = ACFF(16*state_dim+graph_feat_size, emb_size, action_dim, mode='soft')
            self.critic = ACFF(16*state_dim+graph_feat_size, emb_size, 1, mode='')

        elif mode == 'super':
            self.actor = ACFF_SP(state_dim + graph_feat_size, emb_size, action_dim, ntasks=ntasks, mode='soft')
            self.critic = ACFF_SP(state_dim + graph_feat_size, emb_size, 1, ntasks=ntasks, mode='')
        
        elif mode == 'simple_ff':
            self.actor = ACFF(state_dim, emb_size, action_dim, mode='')  # Don't apply softmax since we now use logits
            self.critic = ACFF(state_dim, emb_size, 1, mode='')

        else:
            self.actor = ACFF(act_feat_sz, emb_size, action_dim, mode='soft')  # earlier: state_dim+graph_feat_size
            self.critic = ACFF(act_feat_sz, emb_size, 1, mode='')
        self.mode = mode

    def reset_lstm(self):
        self.actor.reset_lstm()
        self.critic.reset_lstm()

    def forward(self):
        raise NotImplementedError

    def act_seq(self, state, graph_info):
        state_in = state[0].to(_engine)
        mask = state[1].to(_engine)
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

    def act(self, state, graph_info, node_id, mask): # prev_act, pre_constr):
        # graph = dgl.add_self_loop(graph_info)
        # graph_feat = graph.ndata['feat']
        # for layer in self.graph_model:
        #     graph_feat = layer(graph, graph_feat)
        # graph_feat = self.graph_avg_pool(graph, graph_feat)
        # node_feat = graph_feat[node_id, :]

        # emb = self.graph_model(graph_info).squeeze()
        # print('[INFO] state shape', state.shape)
        # print('[INFO] graph_feat shape', graph_feat.shape)
        # act_in = torch.cat((graph_feat, state), -1)
        # act_in = self.pam_attention(act_in.unsqueeze(0)).squeeze(0) # attention module
        # act_in = self.cam_attention(act_in.unsqueeze(0)).squeeze(0)
        # act_in = self.graph_avg_pool(graph, act_in)

        logits = self.actor(state)
        logits = torch.atleast_2d(logits)
        dist = CategoricalMasked(logits=logits, mask=mask)
        action = dist.sample() # flattened index of a tile slice coord
        action_logprob = dist.log_prob(action)

        # if not self.args.no_tm_constr:
        #     for nd in pre_constr['grp_nodes'][node_id]:
        #         if (prev_act[nd] > -1).all(): # if a grouped node is already placed
        #             act_t = list(np.unravel_index(action.item(), self.device['topology']))
        #             act_t[:2] = prev_act[nd][:2] #copy tile loc
        #             if act_t[2] == prev_act[nd][2]:
        #                 free_spoke = list(range(0, self.device['topology'][2]))
        #                 for p_act in prev_act:
        #                     if p_act[:2] == act_t[:2]:
        #                         free_spoke.remove(p_act[2])
        #                 act_t[2] = random.choice(free_spoke)
        #             action.data = torch.tensor([int(ravel_index(act_t, self.device['topology']).item())], dtype=torch.int64)

        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, graph_info, mask, node_id=None, taskid=None):
        # graph = graph_info[0]
        # graph = dgl.add_self_loop(graph)
        # graph_feat = graph.ndata['feat']
        # for layer in self.graph_model:
        #     graph_feat = layer(graph, graph_feat)

        # graph_feat = self.graph_avg_pool(graph, graph_feat)
        # emb = self.graph_model(graph_info)

        # act_in = torch.cat((graph_feat.broadcast_to(state.shape[0], -1, -1), state), dim=-1)
        # act_in = self.pam_attention(act_in) # attention module
        # act_in = self.cam_attention(act_in)

        # act_in = act_in.reshape(-1, act_in.shape[2])
        # act_in = self.graph_avg_pool(dgl.batch(graph_info), act_in)

        if self.mode == 'super':
            action_probs = self.actor(act_in, taskid)
        else:
            logits = self.actor(state)
            logits = torch.atleast_2d(logits)

        dist = CategoricalMasked(logits=logits, mask=mask)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        if self.mode == 'super':
            state_values = self.critic(act_in, taskid)
        else:
            state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy
