import dgl
import torch
import networkx as nx

from dgl import nn as gnn
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from matplotlib import pyplot as plt


def positional_encoding(idx, num_feat=16, timescale=10000):
    '''
    idx: [N x D] matrix of positions
    returned encoding will be [N x D*num_feat]
    '''
    N, D = idx.shape

    sin_freq = torch.arange(0, num_feat, 2.0) / num_feat
    cos_freq = torch.arange(1, num_feat, 2.0) / num_feat

    sin_freq = 1 / (timescale ** sin_freq)
    cos_freq = 1 / (timescale ** cos_freq)

    sin_emb = torch.sin(torch.einsum('ni,d->ndi', idx, sin_freq))
    cos_emb = torch.cos(torch.einsum('ni,d->ndi', idx, cos_freq))

    enc = torch.zeros(N, D*num_feat)
    for i in range(D):
        enc[:, i*num_feat:(i+1)*num_feat:2] = sin_emb[:, :, i]
        enc[:, i*num_feat+1:(i+1)*num_feat:2] = cos_emb[:, :, i]

    return enc

'''
Create Graph
------------

- Create a graph for sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
- This will be our Proof-of-Concept input till we write module to generate
  random graphs
'''
def create_graph():

    src_ids = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 9]).int()
    dst_ids = torch.Tensor([3, 4, 5, 6, 6, 7, 7, 8, 0, 1, 2]).int()
    g = dgl.graph((src_ids, dst_ids))

    node_coords = torch.zeros(g.num_nodes(), 2)

    asc = dgl.topological_nodes_generator(g)
    dsc = dgl.topological_nodes_generator(g, True)

    for i, (nodes_a, nodes_d) in enumerate(zip(asc, dsc)):
        node_coords[nodes_a.long(), 0] = i
        node_coords[nodes_d.long(), 1] = -i
    #node_coords[:, 2] = torch.arange(g.num_nodes())

    encoding = positional_encoding(node_coords, 16)
    rand_feat = encoding.clone().detach()
    rand_feat = rand_feat.uniform_()
    g.ndata['feat'] = encoding + rand_feat

    return g


'''
g = create_graph()
nx_g = g.to_networkx()
nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'))
plt.show()
'''


'''
Create Model
------------

Parameters: Tile Shape [Rows, Cols, Spokes]
Input: Compute Graph
'''

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
#
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
                 sinkhorn_iters=100,
                 device_topology=(4, 4, 3)):

        super().__init__()

        self.sinkhorn_iters = sinkhorn_iters

        self.cg_conv = nn.ModuleList([
            gnn.SGConv(cg_in_feats, cg_hidden_dim, cg_conv_k, False, nn.ReLU),
            gnn.SGConv(cg_hidden_dim, cg_hidden_dim, cg_conv_k, False, nn.ReLU),
            gnn.SGConv(cg_hidden_dim, cg_hidden_dim, cg_conv_k, False, nn.ReLU),
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

        coords = torch.meshgrid(*[torch.arange(i) for i in device_topology])
        coords = [coord.unsqueeze(-1) for coord in coords]
        coords = torch.cat(coords, -1)
        coords = coords.view(-1, coords.shape[-1])

        device_feats = positional_encoding(coords, 32, 100)

        self.coords = coords
        self.device_feats = device_feats.unsqueeze(1)

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter("bin_score", bin_score)

    def forward(self, graph):

        g = dgl.add_self_loop(graph)

        feat = g.ndata['feat']
        for layer in self.cg_conv:
            feat = layer(g, feat)

        cg_feat = feat.unsqueeze(1)

        match_feat = self.match_transfomer(self.device_feats, cg_feat)
        match_feat = self.match_ff(match_feat)

        dist = cg_feat[:, None] - match_feat[None]
        dist = (dist.abs()).sum(-1).permute(2, 0, 1)
        cost_matrix = dist

        #cost_matrix = torch.einsum('ibd,jbd->ibj', cg_feat, match_feat)
        #cost_matrix = cost_matrix.permute(1, 0, 2)
        #cost_matrix = cost_matrix.abs() / feat.shape[-1]**0.5

        scores = self.log_optimal_transport(cost_matrix, self.bin_score)

        if ep%200 == 0:
            plt.imshow(scores[0].detach(), vmin=0, vmax=1)
            plt.xticks(range(len(self.coords)),
                       [f'{x},{y},{z}' for x, y, z in self.coords],
                       rotation=90, fontsize=6)
            plt.pause(1e-6)

        '''
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :].max(2), scores[:, :-1, :].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > 0.1)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        result = []
        for i, coord in enumerate(self.coords):
            op = indices1[0, i].item()
            prob = scores[0, indices1[0, i], i]
            result.append((coord, op, prob))

        return result, scores
        '''

        topk = torch.topk(scores[0, :-1, :], len(self.coords), -1)
        topk = topk.indices

        choices = []
        result = []
        for i in range(topk.shape[0]):
            idx = 0
            for choice in topk[i]:
                if choice in choices: continue
                break
            else:
                result.append((None, -1, None))
                continue

            coord = self.coords[choice]
            result.append((coord, i, scores[0, i, choice]))
            choices.append(choice)

        return result, scores


        with torch.no_grad():

            p = Categorical(probs=scores)
            choice = p.sample()
            choice = choice[0]

            output, inv_idx, counts = torch.unique(choice, True, True, True, dim=-1)
            repeated_choice = output[counts > 1]
            for i in range(10):
                if repeated_choice.numel() == 0: break
                p = Categorical(probs=scores[0, choice == repeated_choice[0], :])
                sub_choice = p.sample()
                choice[choice == repeated_choice[0]] = sub_choice
                output, inv_idx, counts = torch.unique(choice, True, True, True, dim=-1)
                repeated_choice = output[counts > 1]

        result = []
        for i, idx in enumerate(choice):
            op = -1 if idx in repeated_choice else i
            result.append((self.coords[idx], i, scores[0, i, idx]))
        result = result[:-1]

        return result, scores, choice

    def log_optimal_transport(self, cost_matrix, bin_score, lam=1e-3):

        b, m, n = cost_matrix.shape
        one = cost_matrix.new_tensor(1)
        ms, ns = (m*one).to(cost_matrix), (n*one).to(cost_matrix)

        #bins0 = bin_score.expand(b, m, 1)
        bins1 = bin_score.expand(b, 1, n)
        #bins  = bin_score.expand(b, 1, 1)

        #cost_matrix = torch.cat([torch.cat([cost_matrix, bins0], -1),
        #                         torch.cat([bins1, bins], -1)], 1)
        cost_matrix = torch.cat([cost_matrix, bins1], 1)

        norm = - torch.Tensor([lam]).log()

        log_mu = torch.cat([norm.expand(m), (ns - m).log()[None] + norm])
        log_nu = norm.expand(n)#torch.cat([norm.expand(n), ms.log()[None] + norm])

        log_mu = log_mu[None].expand(b, -1)
        log_nu = log_nu[None].expand(b, -1)

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(cost_matrix + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(cost_matrix + u.unsqueeze(2), dim=1)
        cost_matrix += u.unsqueeze(2) + v.unsqueeze(1) - norm

        return torch.exp(cost_matrix)


def calculate_score(g, C1=1.0, C2=1.0, C3=1.0, num_spokes=3):

    # Placement Score --> how many nodes did we map
    placement_score = (g.ndata['coord'].sum(-1) > -3).sum().item()
    placement_score /= g.num_nodes()

    # Timing Score --> how many nodes meet the timing requirement
    timing_score = 0

    # Cycles Score
    ready_time = torch.zeros(len(g.ndata['coord']))

    # go through nodes in topological order
    timing_error = False
    for nodes in dgl.topological_nodes_generator(g):
        if timing_error: break

        for dst in nodes:
            dst_coord = g.ndata['coord'][dst]

            if dst_coord.sum() == -3: break

            dst_ready_time = 0
            for src in g.predecessors(dst):
                src_coord = g.ndata['coord'][src]

                # Time the src coord is ready + travel time
                src_ready_time = ready_time[src].item()
                src_ready_time += (src_coord - dst_coord)[:2].abs().sum()

                if src_ready_time > dst_ready_time:
                    dst_ready_time = src_ready_time

        #    print(dst, dst_coord, dst_ready_time)

            if dst_ready_time == 0:
                ready_time[dst] = 4 + dst_coord[2]
            elif (dst_ready_time % (num_spokes)) == dst_coord[2]:
                # arrive on time
                ready_time[dst] = dst_ready_time + 4
            else:
                timing_error = True
                break

            timing_score += 1

    timing_score /= g.num_nodes()

    score = 0
    score += placement_score * C1

    if placement_score == 1:
        score += timing_score * C2

        if not timing_error:
            cycles = ready_time.max()
            cycles /= g.num_nodes()
            score += C3 * (10 - cycles)

    return score

'''
g = create_graph()
g.ndata['coord'] = torch.Tensor([[1., 0., 1.],
        [2., 0., 2.],
        [3., 2., 2.],
        [2., 3., 0.],
        [3., 1., 2.],
        [3., 1., 0.],
        [3., 2., 0.],
        [1., 3., 0.],
        [0., 3., 1.],
        [3., 0., 2.]])
calculate_score(g)
exit(-1)
'''

model = PolicyNet()

optim = Adam(model.parameters(), lr=1e-4)

beta = 1e-2

from collections import deque

ep = 0
best_score = 0
avg_reward = deque([0], maxlen=64)

while True:

    ep += 1

    g = create_graph()
    assignment, scores = model(g)

    g.ndata['coord'] = -torch.ones(g.num_nodes(), 3)
    pos_logprobs = 0
    pos = 0
    for coord, op, prob in assignment:
        if op == -1: continue
        g.ndata['coord'][op] = coord
        pos_logprobs += prob.log()
        pos += 1

    logp = pos_logprobs / (pos + 1e-6)
    ent = -(scores[:, :-1] * (scores[:, :-1] + 1e-9).log()).mean()

    '''
    matched_ops = (g.ndata['coord'].sum(-1) > -3).sum().item()
    reward = matched_ops
    if not (g.ndata['coord'].sum(-1) == -3).any():
        reward += -calculate_cycles(g) / 100
    reward = reward
    '''
    reward = calculate_score(g, 10, 10)

    baseline = torch.Tensor(avg_reward).mean() / 1.1

    avg_reward.append(reward)

    #if reward - baseline <= 0:
    #    print(ep, reward, baseline.item())
    #    continue

    optim.zero_grad()
    loss = -logp * (reward - baseline) + beta * ent
    loss.backward()
    optim.step()

    print('%d, %.2e, %.2e, %.2e, %.2e, %.2e'%(ep, reward, loss.item(), ent, best_score, torch.Tensor(avg_reward).mean()))

    if reward >= best_score:
        best_score = reward
        print(g.ndata['coord'])

'''
nx_g = g.to_networkx()
nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'))
plt.show()
'''
