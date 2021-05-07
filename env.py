import dgl
import torch
import networkx as nx

from matplotlib import pyplot as plt

from util import positional_encoding

class StreamingEngineEnv:

    '''
    Not a gym env. But has similar API

    device_topology: [Rows, Cols, Spokes] for a given streaming engine setup
    '''

    def __init__(self,
                 device_topology=(4, 4, 3), device_feat_size=48,
                 compute_graph_def=None, graph_feat_size=32):

        # Represent the streaming engine as a vector of positional encodings
        coords = torch.meshgrid(*[torch.arange(i) for i in device_topology])
        coords = [coord.unsqueeze(-1) for coord in coords]
        coords = torch.cat(coords, -1)
        coords = coords.view(-1, coords.shape[-1])

        assert device_feat_size % len(device_topology) == 0, '\
        device_feat_size must be a multiple of device topology dimension'

        assert graph_feat_size % 2 == 0, 'graph_feat_size must be a \
        multiple of 2'

        feat_size = device_feat_size // len(device_topology)
        device_encoding = positional_encoding(coords, feat_size, 1000)

        # TODO: Make compute_graph_def a text file and load it here
        if compute_graph_def is None: raise NotImplementedError

        self.compute_graph_def = compute_graph_def
        self.graph_feat_size = graph_feat_size

        self.coords = coords
        self.device_topology = device_topology
        self.device_encoding = device_encoding
        self.compute_graph = None

        self._gen_compute_graph()

    def _gen_compute_graph(self):

        generator = torch.Generator()

        # TODO: Allow None for compute_graph_def. Generate a random graph
        if self.compute_graph_def is None:
            raise NotImplementedError
        else:
            src_ids, dst_ids = self.compute_graph_def

            src_ids = torch.Tensor(src_ids).int()
            dst_ids = torch.Tensor(dst_ids).int()
            graph = dgl.graph((src_ids, dst_ids))

            # to get consistent states, but also have a random vector per node
            generator.manual_seed(0)

        # use topological rank and reverse topological rank as feat
        node_coord = torch.zeros(graph.num_nodes(), 2)

        asc = dgl.topological_nodes_generator(graph)
        dsc = dgl.topological_nodes_generator(graph, True)
        for i, (nodes_a, nodes_d) in enumerate(zip(asc, dsc)):
            node_coord[nodes_a.long(), 0] = i
            node_coord[nodes_a.long(), 1] = -i

        feat_size = self.graph_feat_size // 2
        encoding = positional_encoding(node_coord, feat_size // 2, 1000)
        rand_enc = encoding.clone().detach().normal_(generator=generator)

        # Adding random vector to encoding helps distinguish between similar
        # nodes according to a paper. TODO: add paper link
        node_feat = torch.cat([encoding, rand_enc], -1)
        graph.ndata['feat'] = node_feat

        self.compute_graph = graph

    def obs(self):
        return self.compute_graph, self.device_encoding

    def reset(self):
        # Generate a new compute graph
        self._gen_compute_graph()
        return self.obs()

    def render(self, debug=False):
        plt.figure()
        nx_graph = self.compute_graph.to_networkx()
        nx.draw(nx_graph, nx.nx_agraph.graphviz_layout(nx_graph, prog='dot'))
        if debug: plt.show()
        else: plt.pause(1e-3)

    def step(self, assignment : dict):
        '''
        action = list of coordinate idx
        '''
        node_coord = -torch.ones(self.compute_graph.num_nodes(), 3)
        for op_idx, coord_idx in enumerate(assignment):
            if coord_idx == -1: continue
            node_coord[op_idx] = self.coords[coord_idx]

        reward = self._calculate_reward(node_coord)

        return reward

    def _calculate_reward(self, node_coord):

        reward = torch.zeros(self.compute_graph.num_nodes())
        ready_time = torch.zeros(self.compute_graph.num_nodes())

        num_nodes = self.compute_graph.num_nodes()
        max_dist = sum(self.device_topology)

        timing_error = False
        for nodes in dgl.topological_nodes_generator(self.compute_graph):

            # For each node in topological order
            for dst in nodes:

                dst_coord = node_coord[dst]

                # if not placed
                if dst_coord.sum() == -3:
                    ready_time[dst] = -2
                    continue

                # if placed, check for time taken
                dst_ready_time = 0
                for src in self.compute_graph.predecessors(dst):
                    src_coord = node_coord[src]
                    src_done_time = ready_time[src].item()

                    if src_done_time < 0:
                        dst_ready_time = -1
                        break

                    src_done_time += (src_coord - dst_coord)[:2].abs().sum()
                    if src_done_time > dst_ready_time:
                        dst_ready_time = src_done_time

                if dst_ready_time == 0:
                    ready_time[dst] = dst_coord[2] + 4
                elif dst_ready_time == -1:
                    ready_time[dst] = -2
                elif dst_ready_time % self.device_topology[2] == dst_coord[2]:
                    ready_time[dst] = dst_ready_time + 4
                else:
                    ready_time[dst] = -1

        reward[ready_time == -2] = 0
        reward[ready_time == -1] = -1
        reward[ready_time >= 0]  = (max_dist*num_nodes - ready_time[ready_time >= 0])/num_nodes

        if (ready_time >= 0).all():
            print(ready_time, node_coord)

        return reward

if __name__ == "__main__":

    src_ids = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    dst_ids = [1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9]
    compute_graph_def = (src_ids, dst_ids)

    env = StreamingEngineEnv(compute_graph_def=compute_graph_def)
    env.reset()
    env.render()
