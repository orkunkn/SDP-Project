import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx

class GraphEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphEnv, self).__init__()
        self.k1 = 1 # coefficient responsible for increase in indegrees
        self.k2 = 1 # coefficient responsible for getting close to an empty source level, value of k2 is the max reward aka emptied level
        self.h = 0.2 # aka harshness, connected to the level emptying term, ranges between (0,1] , higher harshness causes the rewards to diminish except the max one.
        self.k3 = 1 # coefficient responsible for "if moved state is thin reward is less, else moved state is not thin reward is high"

        self.matrix = matrix
        
        self.action_space = gym.spaces.MultiDiscrete([10, 10])
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(3,), dtype=np.float64)
        
    def step(self, action):
        node_to_move = action[0]
        move_level = action[1]
        self.actions.move_node_to_higher_level(node_to_move, move_level)
        old_alc = self.ALC
        old_arl = self.ARL
        old_air = self.AIR
        self.constructor.calculate_graph_metrics()
        # reward = self.k1*(self.constructor.calculate_total_grandparents(node_to_move) - len(self.node_parents[node_to_move]))+ self.k2*(10**(-self.nodes_per_level[self.levels[node_to_move]]*self.h))  + self.k3*(max(0,self.nodes_per_level[self.levels[node_to_move]+1]-self.ALC)**2)
        reward = 50*(self.ALC - old_alc) + 40*(self.ARL - old_arl) + 30*(self.AIR - old_air) # Temporary reward statement
        info = {'ALC': self.ALC, 'AIR': self.AIR, 'ARL':self.ARL, 'reward':reward, 'node_to_move':node_to_move, 'move_level': move_level}
        observation = np.array([self.AIR, self.ARL, self.ALC])

        return observation, reward, self.done, False, info
        
        
    def reset(self, seed=None):
        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level.
        self.levels = {}

        # A dictionary mapping each node to its parent node.
        self.node_parents = {}

        # A list to keep track of the current state of levels.
        self.state_levels = []

        # A dictionary mapping possible changable levels, their nodes and their indegrees.
        self.state_level_vectors = {}

        # A node counter for every level.
        self.nodes_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        # Number of total nodes in graph
        self.total_nodes = 0

        self.done = False

        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

        self.graph = Graph(self)
        self.constructor = Constructor(self)
        self.actions = Actions(self, self.constructor)
        self.graph.convert_matrix_to_graph()
        self.constructor.calculate_graph_metrics()
        
        self.action_space = gym.spaces.MultiDiscrete([self.total_nodes, 10])
        observation = np.array([self.AIR, self.ARL, self.ALC])
        return observation, {}
    