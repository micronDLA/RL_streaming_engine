import dgl
import math
import os
from numpy.lib.shape_base import tile
import torch
import pprint
import networkx as nx
import random
from matplotlib import pyplot as plt
import numpy as np
from util import positional_encoding, calc_score, initial_fill, output_json, ROW, COL, fix_grid_bins

pp =pprint.PrettyPrinter(indent=2)

# device['topology']
# graphdef['graph']

class PreInput:
    '''
        Prepare the input for RL model
    '''
    def __init__(self, args):
        self.args = args # to select which mode to use

    def pre_input(self, tensor_in):
        '''
        combine or pre-proc inputs
        tensor_in: dict with inputs that changes over steps
        '''
        # concat inputs
        rl_state = torch.cat((tensor_in['state'], tensor_in['node_sel']), axis=1)
        return rl_state


    def pre_graph(self, graph_in, device):
        '''
        combine or pre-proc graph features
        graph_in: dict with graph data
        '''

        # Represent the streaming engine as a vector of positional encodings
        # Generate meshgrid so we can consider all possible assignments for (tile_x, tile_y, spoke)
        tile_coords = torch.meshgrid(*[torch.arange(i) for i in device['topology']])
        tile_coords = [coord.unsqueeze(-1) for coord in tile_coords]
        tile_coords = torch.cat(tile_coords, -1)
        tile_coords = tile_coords.view(-1, tile_coords.shape[-1])
        # Shape: (no_of_tiles * no_of_spokes, 3)
        # tile_coords represents all possible SE slices [tile_x, tile_y, spoke_no]
        # assert device_feat_size % len(device_topology) == 0, '\
        # device_feat_size must be a multiple of device topology dimension'
        assert self.args.graph_feat_size % 2 == 0, 'graph_feat_size must be a multiple of 2'

        feat_size = device['action_dim'] // len(device['topology'])
        device_encoding = positional_encoding(tile_coords, feat_size, 1000)  # Shape: (No of slices, 48)

        generator = torch.Generator()
        # to get consistent states, but also have a random vector per node
        generator.manual_seed(0)

        # use topological rank and reverse topological rank as feat
        node_coord = torch.zeros(graph_in['graph'].num_nodes(), 2)
        asc = dgl.topological_nodes_generator(graph_in['graph'])
        dsc = dgl.topological_nodes_generator(graph_in['graph'], True)
        for i, (nodes_a, nodes_d) in enumerate(zip(asc, dsc)):
            node_coord[nodes_a.long(), 0] = i
            node_coord[nodes_a.long(), 1] = -i
        #use initial placement
        # node_coord = initial_place[:, 0:2]

        feat_size = self.args.graph_feat_size // 2  # TODO: Make this compatible with tile_mem_feat
        encoding = positional_encoding(node_coord, feat_size // 2, 1000)  # Shape: (no_of_graph_nodes, 16)
        rand_enc = encoding.clone().detach().normal_(generator=generator)  # Shape: (no_of_graph_nodes, 16)

        # Adding random vector to encoding helps distinguish between similar
        # nodes. This works pretty well, but maybe other solutions exist?
        node_feat = torch.cat([encoding, rand_enc], -1)  # Shape: (no_of_graph_nodes, 32)
        # Add tile memory feature
        # This is a manual hack right now since in IFFT, number of TM is 14
        # and we have one more value for no TM
        # tile_mem_feat = torch.nn.functional.pad(graph.ndata['tm_req'],(0,1))
        tile_mem_feat = graph_in['graph'].ndata['tm_req']
        node_feat = torch.cat([node_feat, tile_mem_feat], -1)
        graph_in['graph'].ndata['feat'] = node_feat

        return graph_in

    def pre_constr(self, action, graphdef, device):
        '''
        contraint before running RL mapper
        '''
        ret = {}
        gprod = np.prod(device['topology'][:2])
        # pre place nodes in tiles
        if not self.args.no_sf_constr:
            not_used = [ii for ii in range(gprod)]
            for node_id in range(0, self.args.nodes):
                if len(graphdef['graph'].predecessors(node_id)) == 0:
                    place = random.choice(not_used)
                    not_used.remove(place)
                    x, y = np.unravel_index(place, device['topology'][:2])
                    action[node_id] = torch.Tensor([x, y, random.randint(0, 2)])

        # find nodes that must go together because they use same tile mem var
        grp_nodes = None
        if not self.args.no_tm_constr:
            grp_nodes = {}  # node n : list of nodes that goes with node n
            for n in range(self.args.nodes):
                grp = []  # nodes that goes with n
                tmem_n = graphdef['tile_memory_req'][n]  # tile mem var used by n
                for nd, tmem in graphdef['tile_memory_req'].items():  # scan other nodes
                    if nd == n:
                        continue
                    if not set(tmem).isdisjoint(tmem_n):
                        grp.append(nd)
                grp_nodes[n] = grp
        ret['grp_nodes'] = grp_nodes
        return ret, action
