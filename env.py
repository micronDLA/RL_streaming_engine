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
from util import output_json

pp =pprint.PrettyPrinter(indent=2)
class StreamingEngineEnv:
    '''
    Not a gym env. But has similar API
    device_topology: [Rows, Cols, Spokes] for a given streaming engine setup
    '''

    def __init__(self,
                 args,
                 graphdef,
                 device = None,
                 init_place=None,
                 emb_mode='topological',
                 placement_mode=''):

        if device is None:
            device = {'topology': (4, 1, 3), 'action_dim': 12}

        if not args.no_device_cross_connections:
            assert device['topology'][0] == 1 or device['topology'][1] == 1, \
                "Device layout needs to be linear"

        self.device = device
        self.graphdef = graphdef
        self.emb_mode = emb_mode
        self.initial_place = init_place
        self.placement_mode = placement_mode

        self.PIPELINE_DEPTH = 3

        self.no_of_valid_mappings = 0
        self.tile_slice_to_node = {}  # What tile slice has what node
        self.title_used = set()
        self.best_time = float('inf')

        self.tm_idx_total = len(graphdef['tile_memory_map'].keys())
        self.nodes_per_tm = self.get_tm_to_node_mapping()

        self.args = args

    def get_tm_to_node_mapping(self):
        tile_memory_req = self.graphdef['nodes_to_tm']
        nodes_per_tm = {tm_idx:[] for tm_idx in range(1, self.tm_idx_total+1)}
        for instr_idx, tm_idxs in tile_memory_req.items():
            for tm_idx in tm_idxs:
                nodes_per_tm[tm_idx].append(instr_idx)

        return nodes_per_tm

    def reset(self):
        self.tile_slice_to_node = {}
        self.title_used.clear()
        return

    def step(self, assignment : dict):
        '''
        action = list of coordinate idx
        '''
        # assignment is array(num_node, 3)
        return self._calculate_reward(assignment)

    def _calculate_reward(self, node_coords):
        '''
        calc score on grid assignment (dawood) returns reward matrix
        node_coord: [node][coord c,y,x]
        self.compute_graph: dgl
        '''
        num_nodes = self.graphdef['graph'].num_nodes()
        reward = torch.zeros(num_nodes)  # Shape: (no_of_graph_nodes,)
        ready_time = torch.zeros(num_nodes)  # Shape: (no_of_graph_nodes,)
        isvalid = False

        max_dist = sum(self.device['topology'])

        timing_error = False
        for nodes in dgl.topological_nodes_generator(self.graphdef['graph']):

            # For each node in topological order
            for dst in nodes:

                dst_coord = node_coords[dst]

                # if not placed
                if dst_coord.sum() == -3:
                    ready_time[dst] = -2
                    continue

                # if placed, check for time taken
                dst_ready_time = 0
                for src in self.graphdef['graph'].predecessors(dst):
                    src_coord = node_coords[src]  # Coordinate of source node in SE
                    src_done_time = ready_time[src].item()  # Done time of source node

                    if src_done_time < 0: # not ready
                        dst_ready_time = -1
                        break

                    abs_dist = (src_coord - dst_coord)[:2].abs().sum()  # Absolute dist betwn source and destination
                    if not self.args.no_device_cross_connections: # linear representation
                        # _dist = int(math.ceil(abs_dist / 2.)) * 2 - 2
                        # src_done_time += _dist / 2 + _dist + 1
                        # src_done_time += int(abs_dist/2) + abs_dist % 2 - 1
                        if self.args.pass_timing:
                            if abs_dist > 2:
                                src_done_time += 2 + (abs_dist - 2)*2
                            else:
                                src_done_time += abs_dist
                        else:
                            src_done_time += abs_dist

                    else: # grid representation
                        # src_done_time += abs_dist + (abs_dist - 1) * 2
                        # At src_done_time, node is ready to be consumed 1 hop away
                        src_done_time += abs_dist - 1

                    if src_done_time > dst_ready_time: # get largest from all predecessors
                        dst_ready_time = src_done_time  #TODO: Isn't variable dst_ready_time more like dst start time

                tile = dst_coord.numpy().astype(int)
                dst_coord_node = self.tile_slice_to_node.get(tuple(tile), -1)
                if dst_ready_time == 0 and  dst_coord_node in [dst, -1] : # placed fine
                    ready_time[dst] = dst_coord[2] + self.PIPELINE_DEPTH
                    self.tile_slice_to_node[tuple(tile)] = dst
                    self.title_used.add(tuple(tile[:2]))
                elif dst_ready_time == -1: # not placed
                    ready_time[dst] = -2
                elif dst_ready_time % self.device['topology'][2] == dst_coord[2] and dst_coord_node in [dst, -1]: # If ready_time % spoke_count is correct
                    ready_time[dst] = dst_ready_time + self.PIPELINE_DEPTH
                    self.tile_slice_to_node[tuple(tile)] = dst
                    self.title_used.add(tuple(tile[:2]))
                else: # fail place
                    ready_time[dst] = -1

        if not self.args.no_tm_constr:
            # Check if tile memory constraints are satisfied
            # Iterate over TMs and check if nodes associated with it are on same tiles
            for tm_idx, nodes in self.nodes_per_tm.items():
                tile_idx = -1  # Tile idx on which all nodes with same tm_idx req should be scheduled
                nodes_on_same_tile = True  # Are all nodes with same tm_idx req on same tile

                for node in nodes:
                    if node_coords[node].sum() == -3:
                        continue
                    if tile_idx == -1:
                        tile_idx = node_coords[node][0]
                    if tile_idx != node_coords[node][0]:
                        # Two nodes which should be on the same tile
                        # because of tile memory constraints are not
                        nodes_on_same_tile = False
                        break

                if not nodes_on_same_tile:
                    if (ready_time >= 0).all():
                        # This print statement will only show the first TM idx for which
                        # all nodes are not on same tile
                        print(f'[INFO] All nodes placed by network but nodes {nodes} should be on same tile for TM idx {tm_idx} but are not')
                    for node in nodes:
                        if node_coords[node].sum() == -3:
                            continue
                        ready_time[node] = -1  # Set node placement as fail

        # Assign reward based on ready time
        reward[ready_time == -2] = 0 # node not placed
        reward[ready_time == -1] = -1 # node place fail
        reward[ready_time >= 0]  = (max_dist*num_nodes - ready_time[ready_time >= 0])/num_nodes

        # self.compute_graph.ndata['node_coord'] = node_coords

        if (ready_time >= 0).all() and self.best_time > ready_time.max().item():
            # Print possible assignment when all nodes are mapped
            self.best_time = ready_time.max().item()
            print('\nPossible assignment -> best time: {} '.format(self.best_time))
            assignment_list = [f'Instr ID# {node_idx}: {int(t)} | {a}' for node_idx, (t, a) in \
                               enumerate(zip(ready_time, node_coords.int().numpy()))]
            print('Instr ID#  : Ready time | Tile slice')
            pp.pprint(assignment_list)
            # output_json(node_coord.numpy(), out_file_name=f'mappings/mapping_{self.no_of_valid_mappings}')
            sulfix = os.path.splitext(os.path.basename(self.args.input))[0]
            output_json(node_coords.numpy(), out_file_name=f'mappings/mapping_{sulfix}.json')
            self.no_of_valid_mappings += 1
            isvalid = True

        return reward, ready_time, isvalid

if __name__ == "__main__":
    pass

