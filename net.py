import torch
import dgl

from torch import nn
from dgl import nn as gnn
import torch.nn.functional as F
import numpy as np
import math
from util import positional_encoding

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1

class PolicyNet(nn.Module):

    '''
    Params
    ------
    cg_* : Params for Compute Graph GNN layers
    device_topology : [Rows, Cols, Spokes]
    '''

    def __init__(self,
                 cg_in_feats=32, cg_hidden_dim=64, cg_conv_k=1,
                 transformer_dim=96, transformer_nhead=4, transformer_ffdim=256,
                 transformer_dropout=0.2, transformer_num_layers=4,
                 sinkhorn_iters=100):

        super().__init__()

        self.sinkhorn_iters = sinkhorn_iters

        self.cg_conv = nn.ModuleList([
            gnn.SGConv(cg_in_feats, cg_hidden_dim, cg_conv_k, False, nn.ReLU),
            gnn.SGConv(cg_hidden_dim, transformer_dim, cg_conv_k)
        ])

        self.match_transfomer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                transformer_dim, transformer_nhead,
                transformer_ffdim, transformer_dropout, 'relu'
            ),
            num_layers=transformer_num_layers
        )
        self.match_ff = nn.Linear(transformer_dim, transformer_dim)

        bin_score = nn.Parameter(torch.tensor(0.))
        self.register_parameter("bin_score", bin_score)

    def forward(self, graph, device_feats, debug=False):

        g = dgl.add_self_loop(graph)

        feat = g.ndata['feat']
        for layer in self.cg_conv:
            feat = layer(g, feat)

        cg_feat = feat.unsqueeze(1)
        device_feats = device_feats.unsqueeze(1)

        match_feat = self.match_transfomer(device_feats, cg_feat)
        match_feat = self.match_ff(match_feat)

        cost_matrix = torch.einsum('ibd,jbd->ibj', cg_feat, match_feat)
        cost_matrix = cost_matrix.permute(1, 0, 2)
        cost_matrix = cost_matrix.abs() / feat.shape[-1]**0.5

        # sinkhorn, returns log(scores)
        scores = self.log_optimal_transport(cost_matrix, self.bin_score)

        # Get the matches with score above "match_threshold"
        # taken from code for SueprGlue paper
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 >= 0)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # TODO: refactor for batches later

        assignment = []
        logp = []
        for i, j in enumerate(indices0[0]):
            if j == -1:
                topk = torch.topk(scores[0, i, :-1].exp(), len(device_feats), -1)
                topk = topk.indices
                for choice in topk:
                    if choice in indices0[0]: continue
                    if choice in assignment: continue
                    break
                else:
                    raise Exception
                logp.append(scores[0, i, choice])
                assignment.append(choice)
            else:
                logp.append(scores[0, i, j])
                assignment.append(j)

        '''
        topk = torch.topk(scores[0, :-1, :].exp(), scores.shape[-1], -1)
        topk = topk.indices

        assignment = []
        logp = []
        for i in range(topk.shape[0]):
            idx = 0
            for choice in topk[i]:
                if choice in assignment: continue
                break
            else:
                raise Exception

            logp.append(scores[0, i, choice])
            assignment.append(choice)
        '''

        entropy = (scores[:, :-1, :-1] * scores[:, :-1, :-1].exp()).sum(-1)
        entropy = entropy[0]

        return assignment, torch.stack(logp), entropy, scores


    def log_optimal_transport(self, cost_matrix, bin_score, lam=1e-1):

        b, m, n = cost_matrix.shape
        one = cost_matrix.new_tensor(1)
        ms, ns = (m*one).to(cost_matrix), (n*one).to(cost_matrix)

        bins0 = bin_score.expand(b, m, 1)
        bins1 = bin_score.expand(b, 1, n)
        bins  = bin_score.expand(b, 1, 1)

        cost_matrix = torch.cat([torch.cat([cost_matrix, bins0], -1),
                                 torch.cat([bins1, bins], -1)], 1)
        #cost_matrix = torch.cat([cost_matrix, bins1], 1)
        cost_matrix = cost_matrix / lam

        norm = - (ms + ns).log()

        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        #log_nu = norm.expand(n)
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])

        log_mu = log_mu[None].expand(b, -1)
        log_nu = log_nu[None].expand(b, -1)

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(cost_matrix + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(cost_matrix + u.unsqueeze(2), dim=1)
        cost_matrix += u.unsqueeze(2) + v.unsqueeze(1) - norm

        return cost_matrix


class GraphNet(nn.Module):
    def __init__(self, out_size, dropout=0.2):
        super(GraphNet, self).__init__()
        graph_emb = 64
        self.cg_conv = nn.Conv1d(2, graph_emb, 1) #2: g.edges src node, dst node
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(graph_emb)

        #class
        self.l1 = nn.Linear(graph_emb, graph_emb)
        self.l2 = nn.Linear(graph_emb, graph_emb)
        self.lo = nn.Linear(graph_emb, out_size)

    def forward(self, x):
        x = self.cg_conv(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(self.norm(x))

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.lo(x)

    def train_step(self, model, train_loader, criterion, optimizer, train=True, device='cpu'):
        self.to(device)
        if train:
            model.train()
        else:
            model.eval()
        train_loss = 0
        for _, (sample, target) in enumerate(train_loader):
            input_tensor = sample.to(device) # (bs, 2, edges)
            target = target.to(device) # (bs, nodes)
            optimizer.zero_grad()
            predictions = self.forward(input_tensor)
            print('predictions: ', predictions, 'target: ', target)
            loss = criterion(predictions, target)  # compare to next values
            if train:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)


class NormalHashLinear(nn.Module): #from briancheung/superposition
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(NormalHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        o = torch.from_numpy(np.random.randn(n_in, period).astype(np.float32))

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)]
        m = x*o
        r = torch.mm(m, self.w)
        return r


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

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.tanh = nn.Tanh()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.tanh(self.encoder(src)) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output