import gym
import dgl
from gym import spaces

import numpy as np
import logging

class Tile:
    """class for a Tile """
    def __init__(self, index, spoke_count):
        self.index = index
        self.spoke_count = spoke_count
        self.spokes = [None] * self.spoke_count

    def place(self, node, spoke_idx):
        assert spoke_idx >=0 and spoke_idx < self.spoke_count, f"Spoke index needs to be in range [0,{self.spoke_count}])"
        assert self.spokes[spoke_idx] == None, f"Can't place node {node} at [Tile, Spoke]: [{self.spokes[spoke_idx]}, {self.index}]. Node {self.spokes[spoke_idx]} previously placed here"
        self.spokes[spoke_idx] = node
        return True  # return True for successful placement

    def reset(self):
        self.spokes = [None] * self.spoke_count

class StreamingEngine:
    def __init__(self, tile_count=16, spoke_count=3, pipeline_depth=3):
        self.tile_count = tile_count
        self.spoke_count = spoke_count
        self.pipeline_depth = pipeline_depth
        self.tiles = [Tile(idx, spoke_count) for idx in range(tile_count)]

    def get_state(self, view=''):
        """Returns array with node_idx where node has been placed, -1 otherwise"""
        state = np.ones(self.tile_count * self.spoke_count) * -1
        for tile_idx, tile in enumerate(self.tiles):
            for spoke_idx, spoke in enumerate(tile.spokes):
                if spoke != None:
                    state[tile_idx*self.spoke_count + spoke_idx] = self.tiles[tile_idx].spokes[spoke_idx]  # or can be 1

        if view == 'human':
            state = state.reshape(self.tile_count, self.spoke_count)
        return state

    def reset(self):
        for tile in self.tiles:
            tile.reset()

class StreamingEngineEnv(gym.Env):
    """Streaming engine class"""
    def __init__(self, args, graphdef=None, tile_count=16, spoke_count=3, pipeline_depth=3):
        super(StreamingEngineEnv, self).__init__()
        self.se = StreamingEngine(tile_count=tile_count, 
                                  spoke_count=spoke_count, 
                                  pipeline_depth=pipeline_depth)
        self.graphdef = graphdef
        self.num_nodes = graphdef['graph'].num_nodes()
        # Action: [Node_idx, tile_idx, spoke_idx]
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.se.tile_count, self.se.spoke_count])
        # Observation: Vector containing info about each tile slice
        self.observation_space = spaces.Discrete(self.se.tile_count * self.se.spoke_count)
        self.placed_nodes = {}  # Keys: node_idx, values: [(tile_idx, spoke_idx]), ready_time]


    def step(self, action):
        node, tile_idx, spoke_idx = action
        assert tile_idx >=0 and tile_idx < self.se.tile_count, f"Tile index not in range [0, {self.se.tile_count}]"
        
        placed = False
        if not self._predecessors_placed(node):
            raise ValueError(f'All predecessors of node {node} not placed')
        if self.placed_nodes.get(node) == None:  # Check if node hasn't been placed already
            placed = self.se.tiles[tile_idx].place(node, spoke_idx)
        else:
            raise ValueError(f'Node {node} already placed at [Tile, Spoke]: {self.placed_nodes.get(node)}')
        if placed:  # `placed ` flag checks if no other node was already present at tile slice
            ready_time, predecessor_ready_time = self._get_ready_time(action)
            self.placed_nodes[node] = {'tile_slice': (tile_idx, spoke_idx), 'ready_time': ready_time}
        obs = self.se.get_state()  # Can change to boolean obs
        reward = self._calculate_reward(ready_time, predecessor_ready_time)
        done = len(self.placed_nodes) == self.num_nodes
        return obs, reward, done, {}

    def reset(self):
        self.se.reset()
        return self.se.get_state()

    def render(self):
        pass

    def _get_ready_time(self, action):
        # Assumes that node has already been placed, along with its predecessors
        node, tile_idx, spoke_idx = action
        predecessors = self.graphdef['graph'].predecessors(node).numpy()
        ready_time = 0
        predecessor_ready_time = 0
        # If node doesn't have any predecessor, processing starts immediately
        if len(predecessors) == 0:
            ready_time = spoke_idx + self.se.pipeline_depth

        else:
            # Get ready greatest ready time of predecessor
            node_predecessor, predecessor_ready_time = -1, -1
            for predecessor in predecessors:
                if self.placed_nodes[predecessor]['ready_time'] > ready_time:
                    predecessor_ready_time = self.placed_nodes[predecessor]['ready_time']
                    node_predecessor = predecessor

            predecessor_tile_idx, predecessor_spoke_idx = self.placed_nodes[predecessor]['tile_slice']
            abs_dist = abs(predecessor_tile_idx - tile_idx)
            ready_time = predecessor_ready_time + abs_dist + self.se.pipeline_depth

        return ready_time, predecessor_ready_time

    def _predecessors_placed(self, node: dgl.DGLGraph.nodes):
        """Check if predecessors of node have been placed

        Args:
            node (dgl.DGLGraph.nodes): The nodes whose predecessors we are checking

        Returns:
            predecessors_placed: True if predecessors have been placed for `node`, False otherwise
        """
        predecessors_placed = True  # Assume predecessors_placed
        # Check if predecessors_placed
        for predecessor in self.graphdef['graph'].predecessors(node).numpy():
            if self.placed_nodes.get(predecessor) == None:
                predecessors_placed = False
                break
        return predecessors_placed

    def get_mask(self, node):
        """Return boolean mask of feasible tile slice locations given node to place
        """
        zero_mask = np.zeros(self.se.tile_count * self.se.spoke_count)
        # If node is already placed, return mask with all zeros
        if self.placed_nodes.get(node) != None:
            logging.debug(f'Node {node} already placed, zero mask returned')
            return zero_mask

        # Check if predecessors have been placed
        elif not self._predecessors_placed(node):
            logging.debug(f'All predecessors not placed for node {node}, zero mask returned')
            return zero_mask
        
        # Node can be placed at any open tile slice if no constraint is applied
        mask = 1 - (self.se.get_state() > -1).astype(int)
        
        if not self.args.no_tm_constr:
            # TODO: Implement tm constr mask
            pass

        if not self.args.no_sf_constr:
            # TODO: Implement sf constr mask
            pass

        return mask


    def _calculate_reward(self, ready_time, predecessor_ready_time):
        # Ready time of node - ready time of parent
        reward = ready_time - predecessor_ready_time
        return reward