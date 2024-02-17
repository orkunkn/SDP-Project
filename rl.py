import gymnasium as gym
from mtx_to_array import mtx_to_array
from environment import Environment
import numpy as np
from stable_baselines3 import PPO
from graph import Graph
from constructor import Constructor
from actions import Actions
from callback import PrintInfoCallback

class GraphOptimizationEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphOptimizationEnv, self).__init__()
        self.k1 = 1 # coefficient responsible for increase in indegrees
        self.k2 = 1 # coefficient responsible for getting close to an empty source level, value of k2 is the max reward aka emptied level
        self.h = 0.2 # aka harshness, connected to the level emptying term, ranges between (0,1] , higher harshness causes the rewards to diminish except the max one.
        self.k3 = 1 # coefficient responsible for "if moved state is thin reward is less, else moved state is not thin reward is high" 
    
        self.env = Environment(matrix)
       
        self.matrix = matrix

        self.graph = Graph(self.env)
        self.constructor = Constructor(self.env)
        self.actions = Actions(self.env, self.constructor)
        self.action_to_node_mapping = {}
        self.graph.convert_matrix_to_graph()
        self.constructor.calculate_graph_metrics()
        
        self.action_space = gym.spaces.Discrete(1000)
        self.update_action_space() # Determine action space
        self.observation_space = gym.spaces.Box(low=0, high=matrix.shape[0], shape=(matrix.shape[0],), dtype=np.float32)

        
    def step(self, action):

        # Check if the action_to_node_mapping is empty, ending the episode if it is
        if not self.action_to_node_mapping:
            done = True
            reward = 0
            info = {'reason': 'no_valid_actions'}
            return self.state, reward, done, False, info


        node_to_move = self.action_to_node_mapping[action]
        self.actions.move_node_to_higher_level(node_to_move)
        done = False
        # reward = self.k1(self.constructor.calculate_total_grandparents(node_to_move) - self.env.node_parents[node_to_move].length)+ self.k2(10**(-self.env.nodes_per_level[self.env.levels[node_to_move]]*self.h))  + self.k3(max(0,self.env.nodes_per_level[self.env.levels[node_to_move]+1]-self.env.ALC)**2)
        reward = 10
        self.state = self._next_state()  # Update the state
        info = {'ALC': self.env.ALC, 'AIR': self.env.AIR, 'ARL':self.env.ARL}

        return self.state, reward, done, False, info

    

    def update_action_space(self):
        # Calculate the number of nodes per level
        nodes_count_per_level = {level: len(nodes) for level, nodes in self.env.state_level_vectors.items()}
        # Sort levels by the number of nodes (ascending order to prioritize thin levels)
        sorted_levels = sorted(nodes_count_per_level, key=nodes_count_per_level.get)
        
        # Flatten the list of nodes from the sorted levels to define the new action space
        action_space_nodes = [node for level in sorted_levels for node in self.env.state_level_vectors[level]]
        # The total number of unique actions corresponds to the number of nodes
        total_actions = len(action_space_nodes)

        # Update the RL environment's action space to a Discrete space with total_actions
        self.action_space = gym.spaces.Discrete(total_actions)

        # Update action to node mapping for RL step function
        self.action_to_node_mapping = {i: node for i, node in enumerate(action_space_nodes)}
        
        
    def reset(self, seed=None):
        # Reset the environment to its initial state
        self.env = Environment(self.matrix)
        self.state = self._initial_state()  # obtain the initial state
        return self.state, {}

    
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
        self.graph.draw_graph(name="first", levels=self.env.levels)


matrix = mtx_to_array("arc130.mtx")
env = GraphOptimizationEnv(matrix)
model = PPO("MlpPolicy", env, verbose=1)

callback = PrintInfoCallback()
model.learn(total_timesteps=10, callback=callback)

