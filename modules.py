import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch import einsum
from torch.distributions import Categorical # discrete
from net import NormalHashLinear, TransformerModel
from dgl import nn as gnn
from util import ravel_index
from einops import reduce
import numpy as np
import math
from typing import Optional

_engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.graphs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []
        self.node_ids = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.graphs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
        del self.node_ids[:]

# From https://boring-guy.sh/posts/masking-rl/
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncode(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncode, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncode, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerAttentionModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout=0.5):
        super(TransformerAttentionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder_layer1 = TransformerEncode(ninp, nhead, nhid, dropout)
        self.encoder_layer2 = TransformerEncode(ninp, nhead, nhid, dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        tmp, attn = self.encoder_layer1(src)
        output, _ = self.encoder_layer1(tmp)
        return output, attn


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

class ActorCritic(nn.Module):
    def __init__(self,
                 args,
                 device,
                 state_dim,
                 emb_size,
                 action_dim,
                 graph_feat_size,
                 gnn_in,
                 ntasks = 1):
        super(ActorCritic, self).__init__()
        self.args = args
        self.device = device

        self.graph_model = nn.ModuleList([
            gnn.SGConv(gnn_in, 64, 1, False, nn.ReLU),
            gnn.SGConv(64, 128, 1, False, nn.ReLU)
        ])
        self.graph_avg_pool = gnn.AvgPooling()

        #Attention modules
        self.pam_attention = PAM_ModuleM(graph_feat_size)
        self.cam_attention = CAM_ModuleM(graph_feat_size)
        self.transf_atten = TransformerAttentionModel(graph_feat_size, 4, 64)

        if args.nnmode == 'simple_ff':
            self.actor = ACFF(state_dim+1, emb_size, action_dim, mode='')  # Don't apply softmax since we now use logits
            self.critic = ACFF(state_dim+1, emb_size, 1, mode='')  # +1 for node_id

        elif (self.args.nnmode == 'ff_gnn' or
            self.args.nnmode == 'ff_gnn_attention' or
            self.args.nnmode == 'ff_transf_attention'):

            self.actor = ACFF(state_dim+1+graph_feat_size, emb_size, action_dim, mode='')  # Don't apply softmax since we now use logits
            self.critic = ACFF(state_dim+1+graph_feat_size, emb_size, 1, mode='')  # +1 for node_id

        else:
            self.actor = ACFF(state_dim+graph_feat_size, emb_size, action_dim, mode='soft')  # earlier: state_dim+graph_feat_size
            self.critic = ACFF(state_dim+graph_feat_size, emb_size, 1, mode='')

    def forward(self):
        raise NotImplementedError

    def act(self, state, graph_info, node_id_or_ids, mask):

        state = torch.atleast_2d(state)

        if (self.args.nnmode == 'ff_gnn' or
            self.args.nnmode == 'ff_gnn_attention' or
            self.args.nnmode == 'ff_transf_attention'):

            graph = dgl.add_self_loop(graph_info)
            graph_feat = graph.ndata['feat']
            for layer in self.graph_model:
                graph_feat = layer(graph, graph_feat)

            if self.args.nnmode == 'ff_gnn_attention':
                graph_feat = self.pam_attention(graph_feat.unsqueeze(0)).squeeze(0) # attention module
                # graph_feat = self.cam_attention(graph_feat.unsqueeze(0)).squeeze(0)
            if self.args.nnmode == 'ff_transf_attention':
                graph_feat, attn = self.transf_atten(graph_feat.unsqueeze(1))
                # graph_feat = graph_feat[0]#.squeeze(1)

            graph_feat = self.graph_avg_pool(graph, graph_feat)
            state = torch.cat((state, node_id_or_ids, graph_feat), dim=1)# Add node id and graph embedding
        else:
            state = torch.cat((state, node_id_or_ids), dim=1) # Add node id

        logits = self.actor(state)
        dist = CategoricalMasked(logits=logits, mask=mask)
        action = dist.sample() # flattened index of a tile slice coord
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, graph_info, mask, node_id_or_ids=None):
        state = torch.atleast_2d(state)

        if (self.args.nnmode == 'ff_gnn' or
            self.args.nnmode == 'ff_gnn_attention' or
            self.args.nnmode == 'ff_transf_attention'):

            graph = graph_info[0]
            graph = dgl.add_self_loop(graph)
            graph_feat = graph.ndata['feat']
            for layer in self.graph_model:
                graph_feat = layer(graph, graph_feat)

            if self.args.nnmode == 'ff_gnn_attention':
                graph_feat = self.pam_attention(graph_feat.unsqueeze(0)).squeeze(0) # attention module
                # graph_feat = self.cam_attention(graph_feat.unsqueeze(-1)).squeeze(-1)

            if self.args.nnmode == 'ff_transf_attention':
                graph_feat, attn0, attn1 = self.transf_atten(graph_feat.unsqueeze(1))
                # graph_feat = graph_feat[0]

            graph_feat = self.graph_avg_pool(graph, graph_feat)
            gnn_feat = graph_feat.broadcast_to(state.shape[0], -1)

            state = torch.cat((state, node_id_or_ids, gnn_feat), dim=1)  # Add node id and graph embedding
        else:
            state = torch.cat((state, node_id_or_ids), dim=1) # Add node id

        logits = self.actor(state)

        dist = CategoricalMasked(logits=logits, mask=mask)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy
