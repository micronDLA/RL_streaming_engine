import torch
import dgl
from scipy.spatial import distance
import numpy as np
import random
import networkx as nx

import torch
import math


def positional_encoding(pos, feat_size=16, timescale=10000):
    '''
    pos : [N X D] matrix of positions

    returns a positional encoding of [N x (D * feat_size)]
    '''

    N, D = pos.shape

    sin_freq = torch.arange(0, feat_size, 2.0) / feat_size
    cos_freq = torch.arange(1, feat_size, 2.0) / feat_size
    sin_freq = 1 / (timescale ** sin_freq)
    cos_freq = 1 / (timescale ** cos_freq)

    sin_emb = torch.sin(torch.einsum('ni,d->ndi', pos, sin_freq))
    cos_emb = torch.cos(torch.einsum('ni,d->ndi', pos, cos_freq))

    encoding = torch.zeros(N, D * feat_size)
    for i in range(D):
        start_idx = i * feat_size
        end_idx   = (i + 1) * feat_size
        encoding[:, start_idx:end_idx:2]   = sin_emb[:, :, i]
        encoding[:, start_idx+1:end_idx:2] = cos_emb[:, :, i]

    return encoding


def calc_score_grid(grid, graph, args):
    '''
    calc score on grid assignment returns single score can reuse free process units
    args.nodes, grid_depth, grid_size
    grid: [node][coord c,y,x]
    graph: dgl
    '''
    worst_dist_possible = args.nodes * np.sum((args.grid_depth, args.grid_size, args.grid_size))
    time = 0
    for nodes in dgl.topological_nodes_generator(graph):
        # break conditions - abort and give worst possible time:
        break_1 = len(np.unique(grid, axis=0)) != len(grid)
        break_2 = grid.max(0)[0] > args.grid_depth or grid.max(0)[1] > args.grid_size or grid.max(0)[2] > args.grid_size
        if break_1 or break_2:
            time = worst_dist_possible  # worst score: all nodes traverse entire grid
            break

        maxdist = 0
        for dst in nodes:
            dst_coord = grid[dst]
            src_a = np.array([grid[src] for src in graph.predecessors(dst)])
            if len(src_a) > 0:
                # largest path to reach dst node
                maxdist = distance.cdist(src_a, np.array([dst_coord]), 'cityblock') # manhattan distance
                maxdist = np.max(maxdist) # get largest distance for all nodes in same rank
        # step in the graph: all nodes can ran in parallel if not for data dependence
        time += maxdist

    return time

def calc_score(grid, graph, args):
    '''
    calc score on grid assignment (grid_v6) returns single score
    args.nodes, grid_depth, grid_size
    grid: [node][coord c,y,x]
    graph: dgl
    '''
    worst_dist_possible = args.nodes * np.sum((args.grid_depth, args.grid_size, args.grid_size))
    time = 0
    # Cycles Score
    ready_time = np.zeros(graph.num_nodes())
    # go through nodes in topological order
    timing_error = False
    for nodes in dgl.topological_nodes_generator(graph):
        # break conditions - abort and give worst possible time:
        break_1 = len(np.unique(grid, axis=0)) != len(grid) #make sure all nodes are placed in different spots
        break_2 = grid.max(0)[0] > args.grid_depth or grid.max(0)[1] > args.grid_size or grid.max(0)[2] > args.grid_size
        #make sure coord are valid
        if break_1 or break_2:
            time = worst_dist_possible  # worst score: all nodes traverse entire grid
            break

        maxdist = 0
        src_a = np.array([])
        for dst in nodes:
            dst_coord = grid[dst]
            # src_a = np.array([grid[src] for src in graph.predecessors(dst)])
            dst_ready_time = 0
            for src in graph.predecessors(dst):
                # Time the src coord is ready + travel time
                src_ready_time = ready_time[src].item()
                if src_ready_time > dst_ready_time: #get slowest src
                    src_a = np.array([grid[src]])
                    dst_ready_time = src_ready_time

            if len(src_a) > 0: #there is predecessor
                # largest path to reach dst node
                maxdist = distance.cdist(src_a, np.array([dst_coord]), 'cityblock') # manhattan distance
                maxdist = np.max(maxdist) # get largest distance for all nodes in same rank

            if dst_ready_time == 0:
                ready_time[dst] = 4 + dst_coord[0]
            elif dst_ready_time > 0:
                ready_time[dst] = dst_ready_time + 4
            else:
                timing_error = True
                time = worst_dist_possible  # worst score: all nodes traverse entire grid
                break

        # step in the graph: all nodes can ran in parallel if not for data dependence
        time += maxdist

    return time

def calc_score_1(graph, node_coord, device_topology = (3, 4, 4), device_cross_connections=False):
    '''
    calc score on grid assignment (dawood) returns reward matrix
    node_coord: [node][coord c,y,x]
    graph: dgl
    '''
    reward = torch.zeros(graph.num_nodes())
    ready_time = torch.zeros(graph.num_nodes())

    num_nodes = graph.num_nodes()
    max_dist = sum(device_topology)

    timing_error = False
    for nodes in dgl.topological_nodes_generator(graph):

        # For each node in topological order
        for dst in nodes:

            dst_coord = node_coord[dst]

            # if not placed
            if dst_coord.sum() == -3:
                ready_time[dst] = -2
                continue

            # if placed, check for time taken
            dst_ready_time = 0
            for src in graph.predecessors(dst):
                src_coord = node_coord[src]
                src_done_time = ready_time[src].item()

                if src_done_time < 0: #not ready
                    dst_ready_time = -1
                    break

                abs_dist = (src_coord - dst_coord)[1:3].abs().sum()
                if device_cross_connections:#linear representation
                    _dist = int(math.ceil(abs_dist / 2.)) * 2 - 2
                    src_done_time += _dist / 2 + _dist + 1
                else:#grid representation
                    src_done_time += abs_dist + (abs_dist - 1) * 2

                if src_done_time > dst_ready_time:#get largest from all predecessors
                    dst_ready_time = src_done_time

            if dst_ready_time == 0: # placed fine
                ready_time[dst] = dst_coord[0] + 4
            elif dst_ready_time == -1: # not placed
                ready_time[dst] = -2
            elif dst_ready_time % device_topology[0] == dst_coord[0]: # placed fine
                ready_time[dst] = dst_ready_time + 4
            else: #fail place
                ready_time[dst] = -1

    reward[ready_time == -2] = 0 #node not placed
    reward[ready_time == -1] = -1 #node place fail
    reward[ready_time >= 0]  = (max_dist*num_nodes - ready_time[ready_time >= 0])/num_nodes

    graph.ndata['node_coord'] = node_coord

    if (ready_time >= 0).all():
        print(ready_time, node_coord)

    return reward

def initial_fill(num_nodes, grid_shape, manual = None):
    '''
    fill array random:
    return
      grid: grid with node number in the array
      grid_in: list of node and its placed coord
      manual: list of index coord for each node
    '''
    grid = np.zeros(grid_shape)
    grid_in = []
    gg = np.prod(grid_shape)
    if isinstance(manual, list):
        place = manual
    else:
        place = random.sample(range(gg), num_nodes) # list of unique elements chosen from the population sequence
    for i, idx in enumerate(place):
        c, y, x = np.unravel_index(idx, grid_shape) # index to [coord c, y, x]
        grid[c, y, x] = i+1 #zero is unassigned
        grid_in.append([c, y, x])
    grid_in = np.array(grid_in)
    return grid, grid_in


def inc_coords(grid, inc):
    '''
    grid: grid with node number in the array
    inc: array of increments
    '''
    aa = grid + inc
    aa[aa < 0] = 0
    return aa