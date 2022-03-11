from numpy.lib.shape_base import tile
import torch
import dgl
from scipy.spatial import distance
import numpy as np
import random
import networkx as nx
from collections import deque
import torch
import math
import json

ROW = 2
COL = 2
PIPE_CYCLE = 4

def create_graph(graphdef, numnodes = 10):
    # random generate a directed acyclic graph
    if graphdef is None:
        graphdef = {}
        a = nx.generators.directed.gn_graph(numnodes)
        graph = dgl.from_networkx(a)
    else:
        tile_memory_req = graphdef['nodes_to_tm']
        edges = graphdef['graphdef']
        graph = dgl.graph((torch.Tensor(edges[0]).int(), torch.Tensor(edges[1]).int()))
        if len(edges) == 3 and edges[2] > 0:
            graph.add_nodes(edges[2])
        tm_idx_total = len(graphdef['tile_memory_map'].keys())
        # Add tile memory constraints as features to graph
        tm_req_feat = torch.zeros(graph.num_nodes(), tm_idx_total) # node, tile mem var index binary vector
        for instr_idx, tm_idxs in tile_memory_req.items():
            for tm_idx in tm_idxs:
                tm_req_feat[instr_idx][tm_idx] = 1
        graph.ndata['tm_req'] = tm_req_feat

        # create list of sync start flow nodes
        sf_nodes = []
        for node in graph.nodes():
            if len(graph.predecessors(node).numpy()) == 0:
                sf_nodes.append(node.item())

    graphdef['graph'] = graph
    graphdef['sf_nodes'] = sf_nodes
    return graphdef
    
def positional_encoding(pos, feat_size=16, timescale=10000):
    '''
    pos : [N X D] matrix of positions. N is the number of slices.

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

def get_graph_json(path):
    with open(path) as file:  # Use file to refer to the file object
        data = json.load(file)
        edge_src = []
        edge_dst = []
        tmem_map = {} # tile_mem variable str : int index
        nidx = 0
        for mem in data['TileMemories'].keys():  # give index to each tile mem variable
            tmem_map[mem] = nidx
            nidx += 1
        nidx = 0
        for graph in data['Program']:  # graphs
            offset = nidx
            for node in graph['SyncFlow']:  # nodes
                for edges in node['SEInst']['Successors']:
                    edge_src.append(nidx)
                    edge_dst.append(edges + offset)
                nidx += 1

        extra_node = (nidx-1) - max(max(edge_src), max(edge_dst))
        nidx = 0
        tmem_req = {} # node id : list of tile mem var indexes
        for graph in data['Program']:  # graphs
            for node in graph['SyncFlow']:  # nodes
                l = [] #tile mem var indexes
                for var in node['SEInst']['SEInstUse']:
                    if var in tmem_map:
                        l.append(tmem_map[var])

                tmem_req[nidx] = l
                nidx += 1

        # Create TM to node mappings
        tm_to_nodes = {tm_idx:[] for tm_idx in range(0, len(tmem_map.keys()))}
        for instr_idx, tm_idxs in tmem_req.items():
            for tm_idx in tm_idxs:
                tm_to_nodes[tm_idx].append(instr_idx)
    return {'graphdef': (edge_src, edge_dst, extra_node), 
            # 'tile_memory_req': tmem_req,  # nodes_to_tm
            'nodes_to_tm': tmem_req,
            'tm_to_nodes': tm_to_nodes, 
            'tile_memory_map': tmem_map}

def output_json(placed_nodes, no_of_tiles=16, spoke_count=3 ,out_file_name='mapping.json'):
    """[summary]

    Args:
        instr_coords (np.array): Array w/ shape [Number of slices, 3]
        out_file (str, optional): Output json name. Defaults to 'mapping.json'.
    """    
    data = {}
    #TODO: Change when using variable spoke count for each tile
    num_spokes = [spoke_count for _ in range(no_of_tiles)]
    data['num_tiles'] = no_of_tiles
    data['num_spokes'] = num_spokes
    mappings = [{'tile_id': tile_idx, 'spoke_map': ['' for _ in range(spoke_count)]} for \
                tile_idx in range (no_of_tiles)]
    # Iterate over assignment
    """
    for instr_idx, tile_coord in enumerate(instr_coords):
        tile_idx = int(tile_coord[0])
        spoke_no = int(tile_coord[2])
        mappings[tile_idx]['spoke_map'][spoke_no] = f'instruction ID#{instr_idx}'
    """

    for node_idx, info in placed_nodes.items():
        tile_idx = info['tile_slice'][0]
        spoke_no = info['tile_slice'][1]
        mappings[tile_idx]['spoke_map'][spoke_no] = f'instruction ID#{node_idx}'


    data['mappings'] = mappings
    with open(out_file_name, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def ravel_index(pos, shape):
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si
    return res

def initial_fill(num_nodes, grid_shape, manual = None):
    '''
    fill array random:
    return
      grid: grid with node number in the array
      grid_in: list of node and its placed coord
      manual: list of index coord for each node
    '''
    grid = -np.ones(grid_shape)# -1 is unassigned
    grid_in = []
    gg = np.prod(grid_shape)
    if isinstance(manual, list):
        place = manual
    else:
        place = random.sample(range(gg), num_nodes) # list of unique elements chosen from the population sequence
    for i, idx in enumerate(place):
        x, y, c = np.unravel_index(idx, grid_shape) # index to [coord c, y, x]
        grid[x, y, c] = i
        grid_in.append([x, y, c])
    grid_in = np.array(grid_in)
    return grid, grid_in, place




