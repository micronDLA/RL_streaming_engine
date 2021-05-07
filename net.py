import torch
import dgl

from torch import nn
from dgl import nn as gnn

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

        scores = self.log_optimal_transport(cost_matrix, self.bin_score)

        # Get the matches with score above "match_threshold".
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
