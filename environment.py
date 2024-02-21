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
        
        # A multi discrete action space. [Node, how many levels to move]. This is just a declarement.
        # Initialization is handled in reset function because node number changes according to the matrix
        self.action_space = gym.spaces.MultiDiscrete([10, 10])

        # Observation space contains AIR, ALC, ARL values. Updated in every step.
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(3,), dtype=np.float64)
        
        
    def step(self, action):
        node_to_move = action[0]
        how_many_levels_to_move = action[1]

        # Node and how many levels to move is chosen randomly by agent.
        self.actions.move_node_to_higher_level(node_to_move, how_many_levels_to_move)
        # Metrics are updated after movement.
        self.constructor.calculate_graph_metrics()

        part_1 = self.k1*(self.constructor.calculate_total_grandparents(node_to_move) - len(self.node_parents.get(node_to_move, [])))
        part_2 = self.k2*(10**(-self.nodes_per_level[self.levels[node_to_move]]*self.h))
        part_3 = self.k3*(max(0,self.nodes_per_level[self.levels[node_to_move]]-self.ALC)**2)
        reward = part_1 + part_2 + part_3

        info = {'ALC': self.ALC, 'AIR': self.AIR, 'ARL':self.ARL, 'reward':reward, 'node_to_move':node_to_move, 'move_level': how_many_levels_to_move}
        observation = np.array([self.AIR, self.ARL, self.ALC])

        print(info)

        # Returns observation, reward, done (always False), truncated (unnecessary so always False) and info.
        return observation, reward, self.done, False, info
        

    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None):
        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level.
        self.levels = {}

        # A dictionary mapping each node to its parent node.
        self.node_parents = {}

        # A dictionary mapping possible changable levels, their nodes and their indegrees.
        self.state_level_vectors = {}

        # A node counter for every level.
        self.nodes_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        # Number of total nodes in graph
        self.total_nodes = 0

        # Since there is not a clear ending statement, it is always False
        self.done = False

        # Updated in constructor
        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

        # Used for converting matrix to graph and drawing the graph
        self.graph = Graph(self)
        # Used by agent to required background calculations
        self.constructor = Constructor(self)
        # Used by agent for actions in graph
        self.actions = Actions(self, self.constructor)

        # Since reset is called before every learning process, these functions are done in reset
        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()
        
        # Action and observation spaces are updated according to matrix
        self.action_space = gym.spaces.MultiDiscrete([self.total_nodes, 10])
        observation = np.array([self.AIR, self.ARL, self.ALC])

        # Do not change
        return observation, {}
    