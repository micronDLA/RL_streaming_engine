import dgl
import math
import torch
import pprint
import networkx as nx
import random
from matplotlib import pyplot as plt
import numpy as np
from util import positional_encoding, calc_score, initial_fill, output_json, ROW, COL, fix_grid_bins

pp =pprint.PrettyPrinter(indent=2)
class StreamingEngineEnv:

    '''
    Not a gym env. But has similar API

    device_topology: [Rows, Cols, Spokes] for a given streaming engine setup
    '''

    def __init__(self, compute_graph_def,
                 device_topology=(4, 4, 3), device_cross_connections=False,
                 device_feat_size=48, graph_feat_size=32):

        # Represent the streaming engine as a vector of positional encodings
        # Generate meshgrid so we can consider all possible assignments for (tile_x, tile_y, spoke)
        coords = torch.meshgrid(*[torch.arange(i) for i in device_topology])
        coords = [coord.unsqueeze(-1) for coord in coords]
        coords = torch.cat(coords, -1)
        coords = coords.view(-1, coords.shape[-1])  
        # Shape: (no_of_tiles * no_of_spokes, 3)
        # coords represents all possible SE slices after next operation

        assert device_feat_size % len(device_topology) == 0, '\
        device_feat_size must be a multiple of device topology dimension'

        assert graph_feat_size % 2 == 0, 'graph_feat_size must be a \
        multiple of 2'

        feat_size = device_feat_size // len(device_topology)
        device_encoding = positional_encoding(coords, feat_size, 1000)  # Shape: (No of slices, 48)

        # TODO: Make compute_graph_def a text file and load it here
        if type(compute_graph_def) != tuple: raise NotImplementedError

        if device_cross_connections:
            assert device_topology[0] == 1 or device_topology[1] == 1, \
                "Device layout needs to be linear"

        self.compute_graph_def = compute_graph_def
        self.graph_feat_size = graph_feat_size

        self.coords = coords
        self.device_topology = device_topology
        self.device_cross_connections = device_cross_connections
        self.device_encoding = device_encoding
        self.compute_graph = None
        self.no_of_valid_mappings = 0

        self._gen_compute_graph()

    def _gen_compute_graph(self):

        generator = torch.Generator()

        if self.compute_graph_def is None:
            nodes = random.randrange(10, 30, 5)
            a = nx.generators.directed.gn_graph(nodes)
            graph = dgl.from_networkx(a)
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
        encoding = positional_encoding(node_coord, feat_size // 2, 1000)  # Shape: (no_of_graph_nodes, 16)
        rand_enc = encoding.clone().detach().normal_(generator=generator)  # Shape: (no_of_graph_nodes, 16)

        # Adding random vector to encoding helps distinguish between similar
        # nodes. This works pretty well, but maybe other solutions exist?
        node_feat = torch.cat([encoding, rand_enc], -1)  # Shape: (no_of_graph_nodes, 32)
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
        nx.draw(nx_graph,
                nx.nx_agraph.graphviz_layout(nx_graph, prog='dot'),
                with_labels=True)
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
        '''
        calc score on grid assignment (dawood) returns reward matrix
        node_coord: [node][coord c,y,x]
        self.compute_graph: dgl
        '''
        reward = torch.zeros(self.compute_graph.num_nodes())  # Shape: (no_of_graph_nodes,)
        ready_time = torch.zeros(self.compute_graph.num_nodes())  # Shape: (no_of_graph_nodes,)

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
                    src_coord = node_coord[src]  # Coordinate of source node in SE
                    src_done_time = ready_time[src].item()  # Done time of source node

                    if src_done_time < 0: # not ready
                        dst_ready_time = -1
                        break

                    abs_dist = (src_coord - dst_coord)[:2].abs().sum()  # Absolute dist betwn source and destination
                    if self.device_cross_connections: # linear representation
                        # _dist = int(math.ceil(abs_dist / 2.)) * 2 - 2
                        # src_done_time += _dist / 2 + _dist + 1
                        src_done_time += int(abs_dist/2) + abs_dist % 2 - 1
                    else: # grid representation
                        # src_done_time += abs_dist + (abs_dist - 1) * 2
                        # At src_done_time, node is ready to be consumed 1 hop away
                        src_done_time += abs_dist - 1  

                    if src_done_time > dst_ready_time: # get largest from all predecessors
                        dst_ready_time = src_done_time  #TODO: Isn't variable dst_ready_time more like dst start time

                if dst_ready_time == 0: # placed fine
                    ready_time[dst] = dst_coord[2] + 4
                elif dst_ready_time == -1: # not placed
                    ready_time[dst] = -2
                elif dst_ready_time % self.device_topology[2] == dst_coord[2]: # If ready_time % spoke_count is correct
                    ready_time[dst] = dst_ready_time + 4
                else: # fail place
                    ready_time[dst] = -1

        reward[ready_time == -2] = 0 # node not placed
        reward[ready_time == -1] = -1 # node place fail
        reward[ready_time >= 0]  = (max_dist*num_nodes - ready_time[ready_time >= 0])/num_nodes

        self.compute_graph.ndata['node_coord'] = node_coord

        if (ready_time >= 0).all():
            # Print possible assignment when all nodes are mapped
            print('Possible assignment ->')
            assignment_list = [f'Instr ID# {node_idx}: {int(t)} | {a}' for node_idx, (t, a) in \
                               enumerate(zip(ready_time, node_coord.int().numpy()))]
            print('Instr ID#  : Ready time | Tile slice')
            pp.pprint(assignment_list)
            output_json(node_coord.numpy(), out_file_name=f'mappings/mapping_{self.no_of_valid_mappings}')
            self.no_of_valid_mappings += 1

        return reward


# environment for grid:
class GridEnv():
    def __init__(self, args, grid_init = None, graph = None):
        # args.nodes, grid_depth, grid_size
        self.args = args
        self.grid_shape = (args.nodes, COL, ROW)

        self.state_dim = args.nodes * 3 + args.nodes # actor input
        # concat of node coord and onehot to select which node to place next
        self.action_dim = np.prod(self.grid_shape)

        self.grid = self.grid_init = grid_init
        self.graph = graph
        self.rst_graph = grid_init is None or graph is None #reset will create new graph
        self.reset()

    def reset(self):
        if self.rst_graph:
            self.graph = dgl.from_networkx(nx.generators.directed.gn_graph(self.args.nodes))
            _, self.grid_init, _ = initial_fill(self.args.nodes, self.grid_shape)
            fix_grid_bins(self.grid_init)

        score_test = calc_score(self.grid_init, self.graph)
        self.grid = self.grid_init.copy()
        return self.grid, -score_test

    #action: grid index flattened
    #state: grid [node][coord x,y,z]
    #reward: score
    def step(self, action, node):
        new_coord = list(np.unravel_index(action, self.grid_shape))
        if new_coord in self.grid.tolist(): # if location already has node, swap nodes
            ndt = self.grid.tolist().index(new_coord)# get node in the new coord
            self.grid[ndt] = self.grid[node]
        self.grid[node] = new_coord
        fix_grid_bins(self.grid)
        reward = -calc_score(self.grid, self.graph) #RL maximize reward, we want to minimize time
        return self.grid, reward

if __name__ == "__main__":

    src_ids = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    dst_ids = [1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9]
    compute_graph_def = (src_ids, dst_ids)

    env = StreamingEngineEnv(compute_graph_def=compute_graph_def)
    env.reset()
    env.render()
