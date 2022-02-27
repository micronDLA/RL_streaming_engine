import gym
from gym import spaces

import numpy as np

class Tile:
    """class for a Tile """
    def __init__(self, index, spoke_count):
        self.index = index
        self.spoke_count = spoke_count
        self.spokes = [None] * self.spoke_count

    def place(self, node, spoke_idx):
        assert spoke_idx >=0 and spoke_idx < self.spoke_count, f"Spoke index needs to be in range [0,{self.spoke_count}])"
        assert self.spokes[spoke_idx] == None, f"Node {self.spokes[spoke_idx]} previously placed at at spoke {self.spokes[spoke_idx]}"
        self.spokes[spoke_idx] = node

    def reset(self):
        self.spokes = [None] * self.spoke_count

class StreamingEngine:
    def __init__(self, tile_count=16, spoke_count=3, pipeline_depth=3):
        self.tile_count = tile_count
        self.spoke_count = spoke_count
        self.pipeline_depth = pipeline_depth
        self.tiles = [Tile(idx, spoke_count) for idx in range(tile_count)]

class StreamingEngineEnv(gym.Env):
    """Streaming engine class"""
    def __init__(self, graph=None):
        super(StreamingEngineEnv, self).__init__()
        self.se = StreamingEngine()
        self.graph = graph
        # Action: [Node_idx, tile_idx, spoke_idx]
        # TODO: Change hardcoded value 7 to number of node indices in graph
        self.action_space = spaces.MultiDiscrete([7, self.se.tile_count, self.se.spoke_count])
        # Observation: Vector containing info about each tile slice
        self.observation_space = spaces.Discrete(self.se.tile_count * self.se.spoke_count)


    def step(self, action):
        node, tile_idx, spoke_idx = action 
        self.se.tiles[tile_idx].place(node, spoke_idx)
        obs = self._gen_obs()
        reward = self._get_reward()
        done = False  # Implement done determination
        return obs, reward, done, {}

    def reset(self):
        for tile in self.se.tiles:
            tile.reset()
        return self._gen_obs()

    def render(self):
        pass

    def close(self):
        pass

    def get_mask(self, node):
        """Return mask of feasible tile slice locations given node to place

        Args:
            node (_type_): _description_
        """
        # TODO: Change mask to actually be set of feasible locations
        mask = np.ones(self.se.tile_count * self.se.spoke_count)
        return mask


    def _gen_obs(self):
        state = np.zeros(self.se.tile_count * self.se.spoke_count)
        for tile_idx, tile in enumerate(self.se.tiles):
            for spoke_idx, spoke in enumerate(tile.spokes):
                if spoke != None:
                    state[tile_idx*self.se.spoke_count + spoke_idx] = 1  # or can be node idx: self.se.tiles.spokes[spoked_idx]
        return state

    def _get_reward(self):
        # TODO: Implement reward function
        return 0