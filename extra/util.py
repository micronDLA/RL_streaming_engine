import dgl
from scipy.spatial import distance
import numpy as np
import random
import networkx as nx
import pickle

# grid: [node][coord x,y,z]
# graph: dgl
def calc_score(grid, graph, args, C1=1.0, C2=1.0, C3=1.0, num_spokes=3):
    #args.nodes, grid_depth, grid_size
    worst_dist_possible = args.nodes * np.sum((args.grid_depth, args.grid_size, args.grid_size))
    time = 0
    # Cycles Score
    ready_time = np.zeros(graph.num_nodes())
    # go through nodes in topological order
    timing_error = False
    timing_score = 0
    for nodes in dgl.topological_nodes_generator(graph):
        # break conditions - abort and give worst possible time:
        break_1 = len(np.unique(grid, axis=0)) != len(grid)
        break_2 = grid.max(0)[0] > args.grid_depth or grid.max(0)[1] > args.grid_size or grid.max(0)[2] > args.grid_size
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
                src_a = np.array([grid[src]])
                # Time the src coord is ready + travel time
                src_ready_time = ready_time[src].item()
                if src_ready_time > dst_ready_time:
                    dst_ready_time = src_ready_time

            if len(src_a) > 0:
                # largest path to reach dst node
                maxdist = distance.cdist(src_a, np.array([dst_coord]), 'cityblock') # manhattan distance
                maxdist = np.max(maxdist) # get largest distance for all nodes in same rank

            if dst_ready_time == 0:
                ready_time[dst] = 4 + dst_coord[2]
            elif (dst_ready_time % (num_spokes)) == dst_coord[2]:
                # arrive on time
                ready_time[dst] = dst_ready_time + 4
            else:
                timing_error = True
                time = worst_dist_possible  # worst score: all nodes traverse entire grid
                break

        # step in the graph: all nodes can ran in parallel if not for data dependence
        time += maxdist

    return time


# fill array random:
def initial_fill(num_nodes, grid_shape):
    grid = np.zeros(grid_shape)
    grid_in = []
    gg = np.prod(grid_shape)
    place = random.sample(range(gg), num_nodes) # list of unique elements chosen from the population sequence
    for i, idx in enumerate(place):
        c, y, x = np.unravel_index(idx, grid_shape) # index to [coord x,y,z]
        grid[c, y, x] = i+1
        grid_in.append([c, y, x])
    grid_in = np.array(grid_in)
    return grid, grid_in

def initial_fill_idx(num_nodes, grid_shape):
    grid = np.zeros(grid_shape)
    grid_in = []
    gg = np.prod(grid_shape)
    place = random.sample(range(gg), num_nodes) # list of unique elements chosen from the population sequence
    for i, idx in enumerate(place):
        c, y, x = np.unravel_index(idx, grid_shape) # index to [coord x,y,z]
        grid[c, y, x] = i+1
        grid_in.append([c, y, x])
    grid_in = np.array(grid_in)
    return grid, grid_in, place

def inc_coords(grid, inc):
    aa = grid + inc
    aa[aa < 0] = 0
    return aa

# environment for grid:
class GridEnv():
    def __init__(self, args, grid_init = None, graph = None):
        # args.nodes, grid_depth, grid_size
        self.args = args
        self.grid_shape = (args.grid_depth, args.grid_size, args.grid_size)

        self.state_dim = args.nodes * 3
        # self.state_dim = args.nodes * 3 + args.nodes + args.graph_size # actor input
        self.action_dim = np.prod(self.grid_shape)

        self.grid = self.grid_init = grid_init
        self.graph = graph
        self.rst_graph = grid_init is None or graph is None #reset will create new graph
        self.reset()

    def reset(self):
        if self.rst_graph:
            self.graph = dgl.from_networkx(nx.generators.directed.gn_graph(self.args.nodes))
            _, self.grid_init = initial_fill(self.args.nodes, self.grid_shape)
        score_test = calc_score(self.grid_init, self.graph, self.args)
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
        reward = -calc_score(self.grid, self.graph, self.args) #RL maximize reward, we want to minimize time
        return self.grid, reward

#state: [node] grid's flatten index
#action: where to put node
#rewards: cost
class Q_learn:
    def __init__(self, args, graph):
        self.args = args
        self.board = np.zeros(self.args.nodes)
        self.graph = graph
        self.states = []  # record hashes takes in an episode
        self.lr = args.q_lr
        self.exp_rate = args.exp_rate
        self.decay_gamma = args.decay_gamma
        self.states_value = {}  # state -> value
        self.reset()

    def reset(self):
        self.board[:] = -1
        self.states = []

    # get unique hash of state
    def get_hash(self, st):
        return str(st.flatten())

    #positions: array of avail idx
    #cur_state: [node] [idx]
    def chooseAction(self, positions, cur_state, node, exp=True):
        # choose action with most expected value
        action = 0
        if np.random.uniform(0, 1) <= self.exp_rate and exp: # exploration
            action = np.random.choice(positions)
        else: # exploitation
            value_max = -np.inf
            for p in positions:
                next_board = cur_state.copy()
                next_board[node] = p
                next_boardHash = self.get_hash(next_board)
                value = self.states_value.get(next_boardHash)
                if value is None:
                    value = 0
                # print("value", value)
                if value >= value_max: # do action with max value
                    value_max = value
                    action = p
            # print("{} takes action {}".format(self.name, action))
        return action

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

    # at the end, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    #get not used indexes
    def availablePositions(self):
        aa = [a for a in range(self.args.grid_depth * self.args.grid_size * self.args.grid_size)]
        for i in self.board:
            if i >= 0:
                aa.remove(i)
        return aa

    # get reward and update state_value table
    def giveReward(self):
        grid_in = []
        for i, idx in enumerate(self.board):
            c, y, x = np.unravel_index(int(idx), (self.args.grid_depth, self.args.grid_size, self.args.grid_size))
            grid_in.append([c, y, x])
        rwd = -calc_score(np.array(grid_in), self.graph, self.args)
        self.feedReward(rwd)
        return rwd

    def play(self, rounds=100):
        for i in range(rounds):
            for node in range(self.args.nodes):
                positions = self.availablePositions()
                p_action = self.chooseAction(positions, self.board, node)
                self.board[node] = p_action
                board_hash = self.get_hash(self.board)
                self.states.append(board_hash)

            rwd = self.giveReward()
            if i % 500 == 0:
                print(i, ' reward: ', rwd)
            self.reset()

    def test(self):
        self.reset()
        for node in range(self.args.nodes):
            positions = self.availablePositions()
            p_action = self.chooseAction(positions, self.board, node, exp=False)
            self.board[node] = p_action
        grid_in = []
        for i, idx in enumerate(self.board):
            c, y, x = np.unravel_index(int(idx), (self.args.grid_depth, self.args.grid_size, self.args.grid_size))
            grid_in.append([c, y, x])
        rwd = -calc_score(np.array(grid_in), self.graph, self.args)
        print('reward: ', rwd)

