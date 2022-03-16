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
    # TODO: Implement sibling nodes constraint
    """Streaming engine class"""
    def __init__(self, args, graphdef=None, tile_count=16, spoke_count=3, pipeline_depth=3):
        super(StreamingEngineEnv, self).__init__()
        self.se = StreamingEngine(tile_count=tile_count, 
                                  spoke_count=spoke_count, 
                                  pipeline_depth=pipeline_depth)
        self.args = args
        self.graphdef = graphdef
        self.num_nodes = graphdef['graph'].num_nodes()
        # Action: [Node_idx, tile_idx, spoke_idx]
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.se.tile_count, self.se.spoke_count])
        # Observation: Vector containing info about each tile slice
        self.observation_space = spaces.Discrete(self.se.tile_count * self.se.spoke_count)
        self.placed_nodes = {}  # Keys: node_idx, values: [(tile_idx, spoke_idx]), ready_time]
        self.all_nodes_placed = False
        self.graph_ready_time = -1

    def step(self, action):
        node, tile_idx, spoke_idx = action
        assert tile_idx >=0 and tile_idx < self.se.tile_count, f"Tile index not in range [0, {self.se.tile_count-1}]"
        mask = self.get_mask(node)
        if not mask.any():  # If no action is possible, return high negative reward
            obs = self.se.get_state()
            reward = -10.0
            done = True
            return obs, reward, done, {}
        if not self._predecessors_placed(node):  # Check if predecessors have been placed
            raise ValueError(f'All predecessors of node {node} not placed')
        if self.placed_nodes.get(node) != None:  # Check if node hasn't been placed already
            raise ValueError(f'Node {node} already placed at [Tile, Spoke]: {self.placed_nodes.get(node)}')
        if mask[tile_idx*self.se.spoke_count + spoke_idx] == 0:  # Check if mask allowed node to be placed
            print(f'\nERROR while trying to place: {action}')
            print(f'Currently placed nodes: {self.placed_nodes}', f'Mask: {mask}')
            raise ValueError(f'Illegal placement, action not allowed by mask')
        
        self.se.tiles[tile_idx].place(node, spoke_idx)
        ready_time, predecessor_ready_time = self._get_ready_time(action)
        if ready_time > self.graph_ready_time:  # Keep track of highest ready time of nodes
            self.graph_ready_time = ready_time
        self.placed_nodes[node] = {'tile_slice': (tile_idx, spoke_idx), 'ready_time': ready_time}
        if len(self.placed_nodes) == self.num_nodes:
            self.all_nodes_placed = True
        obs = self.se.get_state()  # Can change to boolean obs
        reward = self._calculate_reward(ready_time, predecessor_ready_time)
        done = len(self.placed_nodes) == self.num_nodes
        return obs, reward, done, {}

    def reset(self):
        self.se.reset()
        self.placed_nodes = {}
        self.all_nodes_placed = False
        self.graph_ready_time = -1
        return self.se.get_state()

    def render(self):
        pass

    def _get_ready_time(self, action):
        # Assumes that node has already been placed, along with its predecessors
        node, tile_idx, spoke_idx = action
        predecessors = self._get_predecessors(node)
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
        for predecessor in self._get_predecessors(node):
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
        
        predecessors = tuple(self._get_predecessors(node))

        # Mask according to timing constraints
        if len(predecessors) != 0:
            latest_pred= max(predecessors, key=lambda predecessor: self.placed_nodes.get(predecessor)['ready_time'])  # Get spoke of predecessor that has latest ready time
            pred_tile = self.placed_nodes.get(latest_pred)['tile_slice'][0]
            pred_spoke = self.placed_nodes.get(latest_pred)['tile_slice'][1]
            for tile_idx in range(self.se.tile_count):
                # Determine spoke idx in each tile which satisfies timing constraint (w/o considering delay)
                avail_spoke_idx_in_tile = (pred_spoke + self.se.pipeline_depth + abs(tile_idx-pred_tile)) % self.se.spoke_count  # Assumes passthrough
                avail_mask_idx = tile_idx * self.se.spoke_count + avail_spoke_idx_in_tile
                unavail_idxs = self._get_spoke_idxs_in_tile(tile_idx)
                unavail_idxs.remove(avail_mask_idx)
                mask[unavail_idxs] = 0

        # predecessor_spoke = self.placed_nodes.get()

        # Mask for sibling constraint
        if not self.args.no_sibling_constr:
            siblings = []
            for predecessor in predecessors:
                successors = self._get_successors(predecessor)
                for successor in successors:
                    if successor != node:
                        siblings.append(successor)

            for sibling in siblings:
                sibling_placement = self.placed_nodes.get(sibling)
                if sibling_placement != None:
                    sibling_tile = sibling_placement['tile_slice'][0]
                    # Make spokes in sibling_tile unavailable
                    unavail_idxs = self._get_spoke_idxs_in_tile(sibling_tile)
                    mask[unavail_idxs] = 0

        # Mask for TM constraint
        if not self.args.no_tm_constr:
            # What TMs does node use
            tms_used = self.graphdef['nodes_to_tm'][node]

            # What other nodes use these TMs?
            other_nodes = set()
            for tm in tms_used:
                for other_node in self.graphdef['tm_to_nodes'][tm]:
                    if other_node != node:
                        other_nodes.add(other_node)
            other_nodes = list(other_nodes)

            # If other nodes are already placed, only the tile they are placed on should be available
            for other_node in other_nodes:
                other_node_placement = self.placed_nodes.get(other_node)
                if other_node_placement != None:
                    other_node_tile = other_node_placement['tile_slice'][0]  # Tile idx
                    # Make every idx in except those in other_node_tile unavailable
                    exculde_idxs = self._get_spoke_idxs_in_tile(other_node_tile)
                    unavail_idxs = [j for j in range(len(mask)) if j not in exculde_idxs]
                    mask[unavail_idxs] = 0

        # Mask for SF constraint
        if not self.args.no_sf_constr:
            predecessors = self._get_predecessors(node)
            if len(predecessors) == 0:
                # Iterate over other sf nodes
                for sf_node in self.graphdef['sf_nodes']:
                    if sf_node != node:
                        sf_node_placement = self.placed_nodes.get(sf_node)
                        if sf_node_placement != None:
                            sf_node_tile = sf_node_placement['tile_slice'][0]
                            # Make spokes in sf_node_tile unavailable
                            unavail_idxs = self._get_spoke_idxs_in_tile(sf_node_tile)
                            mask[unavail_idxs] = 0

        return mask


    def _calculate_reward(self, ready_time, predecessor_ready_time):
        # Ready time of node - ready time of parent
        reward = ready_time - predecessor_ready_time
        return reward

    def _get_predecessors(self, node):
        return self.graphdef['graph'].predecessors(node).numpy()

    def _get_successors(self, node):
        return self.graphdef['graph'].successors(node).numpy()

    def _get_spoke_idxs_in_tile(self, tile_idx):
        idxs = [i for i in range(tile_idx * self.se.spoke_count, tile_idx * self.se.spoke_count + self.se.spoke_count)]
        return idxs