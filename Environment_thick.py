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

        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level.
        self.levels = {}

        # A dictionary mapping each node to its parent node.
        self.node_parents = {}

        # A dictionary mapping possible changable levels, their nodes and their indegrees.
        self.state_level_vectors = {}

        # A node counter for every level.
        self.node_count_per_level = {}

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

        self.threshold = 16

        # Used for converting matrix to graph and drawing the graph
        self.graph = Graph(self)
        self.graph.draw_graph(self)
        # Used by agent to required background calculations
        self.constructor = Constructor(self)
        # Used by agent for actions in graph
        self.actions = Actions(self, self.constructor)

        # Since reset is called before every learning process, these functions are done in reset
        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()
        
        # A discrete action space. Nodes will be chosen.
        # Action and observation spaces are updated according to matrix
        self.action_space = gym.spaces.MultiDiscrete([self.total_nodes, 10])

        # Observation space contains nodes' levels.
        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(self.total_nodes,), dtype=np.int64)
        
        
    def step(self, action):

        # Node is chosen by agent.
        node_to_move = action[0]
        levels_to_drop = action[1]

        # Keep the old values for reward comparison
        old_max_level = max(self.levels.values())
        old_level_node_count = self.node_count_per_level[self.levels[node_to_move] - 1] if self.levels[node_to_move] > 0 else self.node_count_per_level[self.levels[node_to_move]]
        old_level_cost = self.level_costs[self.levels[node_to_move] - 1] if self.levels[node_to_move] > 0 else self.level_costs[self.levels[node_to_move]]

        # Move the node one level upper
        self.actions.move_node_to_higher_level_thick(node_to_move, levels_to_drop)
        # Metrics are updated after movement.
        self.constructor.calculate_graph_metrics()

        """
        part_1 = self.k1*(self.constructor.calculate_total_grandparents(node_to_move) - len(self.node_parents.get(node_to_move, [])))
        part_2 = self.k2*(10**(-self.node_count_per_level[self.levels[node_to_move]]*self.h))
        part_3 = self.k3*(max(0,self.node_count_per_level[self.levels[node_to_move]]-self.ALC)**2)
        reward = part_1 + part_2 + part_3
        """

        reward = self.calculate_reward(node_to_move, old_max_level, old_level_node_count, old_level_cost)
        info = {}

        # Convert object to a list
        data = list(self.levels.values())
        
        # Convert list to an array
        observation = np.array(data)

        # Returns observation, reward, done (always False), truncated (unnecessary so always False) and info.
        return observation, reward, self.done, False, info
        
    def calculate_reward(self, node, old_max_level, old_level_node_count, old_level_cost):
        # New values
        new_max_level = max(self.levels.values())
        new_level_node_count = self.node_count_per_level[self.levels[node]]
        new_level_cost = self.level_costs[self.levels[node]]

        # Level delete reward
        level_deleted_reward = 50*(old_max_level - new_max_level)

        """ Node threshold reward """

        # Reached threshold
        if new_level_node_count == self.threshold:
            threshold_reward = 30

        # Got closer to threshold
        elif abs(old_level_node_count - self.threshold) > abs(new_level_node_count - self.threshold):
            threshold_reward = 15 / abs(new_level_node_count - self.threshold)
        
        # Got further from threshold
        elif abs(old_level_node_count - self.threshold) < abs(new_level_node_count - self.threshold):
            threshold_reward = -5 * abs(new_level_node_count - self.threshold)

        # Rare cases for nodes in level 0
        else:
            threshold_reward = 0

        """ Cost balance reward """
            
        # Level cost is equal to ALC
        if new_level_cost == self.ALC:
            cost_balance_reward = 30

        # Level cost got closer to ALC
        elif abs(new_level_cost - self.ALC) < abs(old_level_cost - self.ALC):
            cost_balance_reward = abs(new_level_cost - self.ALC) / 10

        # Level cost got further from ALC
        else:
            cost_balance_reward = -abs(new_level_cost - self.ALC) / 10
        
        total_reward = level_deleted_reward + threshold_reward + cost_balance_reward

        info = {"level reward":level_deleted_reward, "threshold reward":threshold_reward, "cost balance reward":cost_balance_reward}
        print(info)

        return total_reward

    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None):

        self.constructor.calculate_graph_metrics()
        
        # Convert object to a list
        data = list(self.levels.values())
        
        # Convert list to an array
        observation = np.array(data)

        # Do not change
        return observation, {}
    
    def render(self):
        self.graph.draw_graph()
    