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

def calc_score_0(grid, graph, args):
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
    # a[z,y,x]
    return a[1:3]

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
            if node_level[tuple(dst_coord)] < grid[node][0]:
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

