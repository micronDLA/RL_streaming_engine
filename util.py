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
        a = nx.generators.directed.gn_graph(numnodes)
        graph = dgl.from_networkx(a)
    else:
        tile_memory_req = graphdef['tile_memory_req']
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
    return graph
    
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
    return {'graphdef': (edge_src, edge_dst, extra_node), 'tile_memory_req': tmem_req, 'tile_memory_map': tmem_map}

def output_instr_json(grid_in, grid_shape, filename='output.json'):
    data = {}
    a = np.prod(grid_shape[1:3])
    for idx in range(a):
        y, x = np.unravel_index(idx, grid_shape[1:3])
        data[str((y, x))] = []
    max_spokes = np.amax(grid_in, axis=0)
    for spoke in range(max_spokes[0]):
        for nd_id, nd in enumerate(grid_in):
            if spoke == nd[0]:
                data[str((nd[1], nd[2]))].append(nd_id)
            else:
                data[str((nd[1], nd[2]))].append(-1) # -1 is a nop instr. title is doing nothing
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def output_json(instr_coords, no_of_tiles=16, spoke_count=3 ,out_file_name='mapping.json'):
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
    for instr_idx, tile_coord in enumerate(instr_coords):
        tile_idx = int(tile_coord[0])
        spoke_no = int(tile_coord[2])
        mappings[tile_idx]['spoke_map'][spoke_no] = f'instruction ID#{instr_idx}'

    data['mappings'] = mappings
    with open(out_file_name, 'w') as outfile:
        json.dump(data, outfile, indent=4)

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


def inc_coords(grid, inc):
    '''
    grid: grid with node number in the array
    inc: array of increments
    '''
    aa = grid + inc
    aa[aa < 0] = 0
    return aa



# A data structure for queue used in BFS
class queueNode:
    def __init__(self, pt, dist):
        self.pt = pt  # The coordinates of the cell
        self.dist = dist  # Cell's distance from the source

# Check whether given cell(row,col) is a valid cell or not
def isValid(row: int, col: int):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)


# These arrays are used to get row and column numbers of 4 neighbours of a given cell
rowNum = [-1, 0, 0, 1]  # left, down, up, right
colNum = [0, -1, 1, 0]

def reconstruct(source, dest, lst):
    prev = lst[tuple(dest)]
    ll = []
    ll.append(prev)
    while prev != tuple(source):
        prev = lst[prev]
        ll.append(prev)
    ll.reverse()
    return ll

# Function to find the shortest path between a given source cell to a destination cell.
def BFS(src, dest):
    # src[0] row, src[1] col
    if tuple(src) == tuple(dest):
        return 0, []
    visited = [[False for i in range(COL)] for j in range(ROW)]  # visited matrix
    # Mark the source cell as visited
    visited[src[0]][src[1]] = True
    # Create a queue for BFS
    q = deque()
    # Distance of source cell is 0
    s = queueNode(src, 0)
    q.append(s)  # Enqueue source cell
    lst = {}
    # Do a BFS starting from source cell
    while q:
        curr = q.popleft()  # Dequeue the front cell
        # If we have reached the destination cell, we are done
        pt = curr.pt
        if pt[0] == dest[0] and pt[1] == dest[1]:
            return curr.dist, reconstruct(src, dest, lst)
        # Otherwise enqueue its adjacent cells
        for i in range(4):
            row = pt[0] + rowNum[i]
            col = pt[1] + colNum[i]
            # if adjacent cell is valid, has path and not visited yet, enqueue it.
            if (isValid(row, col) and
                    not visited[row][col]):
                visited[row][col] = True
                Adjcell = queueNode([row, col], curr.dist + 1)
                q.append(Adjcell)
                lst[(row, col)] = (pt[0], pt[1])  # keep track of patent

    # Return -1 if destination cannot be reached
    return -1

def fix_grid_bins(grid_in):
    # sort nodes in the z dimention of grid_in (node order)
    dic = {}
    for i, nd in enumerate(grid_in):
        t = tuple(get_coord(nd)) #get grid coord
        if t in dic.keys():
            grid_in[i][0] = dic[t]
            dic[t] += 1
        else:
            grid_in[i][0] = 0
            dic[t] = 1

def get_coord(a): # get grid coord y, x
    # a[x,y,z]
    return a[0:2]

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def calc_score(grid, graph):
    '''
    calc score on grid assignment returns single score can reuse free process units
    args.nodes, grid_depth, grid_size
    grid: [node][coord c,y,x]
    graph: dgl
    '''
    num_nodes = graph.number_of_nodes()
    placed_node = [nid.item() for nodes in dgl.topological_nodes_generator(graph) for nid in nodes]
    # get all nodes in topological_order
    wait_node = {}
    # node is waiting to receive data {nodeid: [ (src, [path of units to pass]), ... ]  }
    proc_node = {}
    # node is processing {nodeid:wait}
    ready_node = set()
    # node is done {nodeid}
    node_level = np.zeros((COL, ROW))
    # sequence to be placed in the unit. start with 0 goes to num_nodes
    cycle = 0  # cycle count
    done = False
    while not done:

        # check all nodes get distance to wait
        rm_elem = []
        for i, node in enumerate(placed_node):
            # node placed location
            dst_coord = get_coord(grid[node])
            if node_level[tuple(dst_coord)] < grid[node][2]:
                continue  # there is another node to be placed here before

            src_nodes = [src.item() for src in graph.predecessors(node)]
            lst_src = []
            if len(src_nodes) > 0:  # there is a predecessor
                for src in src_nodes:
                    src_coord = get_coord(grid[src])
                    dist, lst = BFS(src_coord, dst_coord)  # get list of coord to pass data
                    lst_src.append((src, lst))

            rm_elem.append(i)
            node_level[tuple(dst_coord)] += 1
            wait_node[node] = lst_src
        #remove placed_node->wait_node
        delete_multiple_element(placed_node, rm_elem)

        # decrement wait
        rm_elem = []
        for k in wait_node.keys():
            cnt = 0
            for i, src in enumerate(wait_node[k]):  # all dependency src
                if len(src[1]) == 0:  # data arrived
                    cnt += 1  # count number of src ready
                elif src[0] in ready_node:  # src is ready
                    del src[1][0]  # pop a data pass

            if len(wait_node[k]) == cnt:  # received all data
                rm_elem.append(k)
                proc_node[k] = PIPE_CYCLE  # how many cycle to get result
        # remove wait_node->proc_node
        for i in rm_elem:
            del wait_node[i]

        # decrement proc_node
        rm_elem = []
        for k in proc_node.keys():
            proc_node[k] -= 1
            if proc_node[k] <= 0:  # done processing
                rm_elem.append(k)
                ready_node.add(k)
        # remove proc_node->ready_node
        for i in rm_elem:
            del proc_node[i]

        cycle += 1
        if len(ready_node) == num_nodes and len(proc_node) == 0:  # no more nodes to process all is ready
            done = True
    return cycle

