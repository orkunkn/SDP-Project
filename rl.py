import gym
from mtx_to_array import mtx_to_array
from environment import Environment
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from graph import Graph
from constructor import Constructor
from actions import Actions

class GraphOptimizationEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphOptimizationEnv, self).__init__()
        self.k1 = 1 # coefficient responsible for increase in indegrees
        self.k2 = 1 # coefficient responsible for getting close to an empty source level, value of k2 is the max reward aka emptied level
        self.h = 0.2 # aka harshness, connected to the level emptying term, ranges between (0,1] , higher harshness causes the rewards to diminish except the max one.
        self.k3 = 1 # coefficient responsible for "if moved state is thin reward is less, else moved state is not thin reward is high" 
        self.action_space = spaces.Discrete(matrix.shape[0])  # Assuming action is to select a node
        self.observation_space = spaces.Box(low=0, high=matrix.shape[0], shape=(matrix.shape[0],), dtype=np.float32)
    
        self.env = Environment(matrix)
        self.matrix = matrix
        
        self.graph = Graph()
        self.constructor = Constructor()
        self.actions = Actions()

        
    def step(self, action):
        # Convert the action index back to the actual node identifier.
        node_to_move = self.action_to_node_mapping[action]
        
        # Perform the action: move the node to a higher level.
        self.actions.move_node_to_higher_level(node_to_move)
        
        done = False  # determine when an episode ends

        reward = self.k1(self.env.calculate_total_grandparents(node_to_move) - self.env.node_parents[node_to_move].length )+ self.k2(10**(-self.env.nodes_per_level[self.env.levels[node_to_move]]*self.h))  + self.k3(max(0,self.env.nodes_per_level[self.env.levels[node_to_move]+1]-self.env.ALC)**2)

        self.state = self._next_state()  # update the state
        return self.state, reward, done, {}
    

    def update_action_space(self):
        eligible_nodes = [node for node, level in self.env.levels.items() if level < max(self.env.levels.values())]

        self.action_space = spaces.Discrete(len(eligible_nodes))
        
        # Keep a mapping of actions to nodes, since the action space is now indexes in the eligible_nodes list.
        self.action_to_node_mapping = {i: node for i, node in enumerate(eligible_nodes)}
        
        
    def reset(self):
        # Reset the environment to its initial state
        self.env = Environment(self.matrix)
        self.state = self._initial_state()  # obtain the initial state
        return self.state

    
    def _next_state(self):
        new_state_representation = np.zeros(self.matrix.shape[0], dtype=np.float32)
        for node, level in self.env.levels.items():
            new_state_representation[node] = level
        self.state = new_state_representation
        return self.state

    
    def _initial_state(self):
        state_representation = np.zeros(self.matrix.shape[0], dtype=np.float32)
        for node, level in self.env.init_levels.items():
            state_representation[node] = level
        self.state = state_representation
        return self.state

        
    def render(self):
        # Optional
        pass


matrix = mtx_to_array("bcsstk17.mtx")
env = GraphOptimizationEnv(matrix)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10)
