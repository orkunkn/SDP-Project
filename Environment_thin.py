import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx

class GraphEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphEnv, self).__init__()

        """ Reward coefficients """
        self.k1 = 50
        self.k2 = 1.5
        self.k3 = 50
        self.h1 = 0.03
        self.h2 = 0.05
        self.h3 = 0.08

        self.matrix = matrix
        
        
    def step(self, action):

        # Node is chosen by agent.
        agent_choice = action

        # Finding the node from nodes in thin level dict
        if agent_choice in self.nodes_in_thin_levels_mapping:
            node_to_move = self.nodes_in_thin_levels_mapping[agent_choice]
        
        # If node is previously moved and not in a thin level anymore
        else:
            reward = -100

            terminated = False

            # Convert object to a list
            data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())
            
            # Convert list to an array
            observation = np.array(data)

            # Returns observation, reward, done (always False), truncated (unnecessary so always False) and info.
            return observation, reward, terminated, False, {}

        # Keep the old values for reward comparison
        old_node_level = self.levels[node_to_move]
        source_node_count = self.levels[node_to_move] - 1

        # Move the node one level upper
        self.actions.move_node_to_higher_level_thin(node_to_move)
        # Metrics are updated after movement.
        self.constructor.calculate_graph_metrics()

        # reward, info = self.calculate_reward(node_to_move, old_max_level, old_level_node_count, old_level_cost)
        reward, info = self.calculate_reward(node_to_move, old_node_level, source_node_count)

        # Learning is done if there is one or zero thin level left.
        terminated = len(self.thin_levels) <= 1

        # Convert object to a list
        data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())
        
        # Convert list to an array
        observation = np.array(data)

        print(info)

        # Returns observation, reward, done (always False), truncated (unnecessary so always False) and info.
        return observation, reward, terminated, False, info
        
    def calculate_reward(self, node, old_node_level, source_node_count):
        new_level_cost = self.level_costs[self.levels[node]]
        part_1, part_2, part_3 = 0, 0, 0

        if old_node_level == self.levels[node]:
            total_reward = -50
        else:
            part_1 = self.k1 * 10**(-source_node_count * self.h1)

            part_2 = -self.k2 * (10**(self.h2 * (old_node_level - self.levels[node])))

            part_3 = self.k3 / (1 + self.h3 * abs(new_level_cost - self.ALC))
            total_reward = part_1 + part_2 + part_3

        # info = {"part 1":part_1, "part 2":part_2, "part 3":part_3, "ALC":self.ALC, "ARL":self.ARL, "done": len(self.thin_levels) <= 1}
        info = {"reward":total_reward, "ALC":self.ALC, "ARL":self.ARL, "AIR":self.AIR}
        return total_reward, info
    

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
        self.node_count_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        # Number of total nodes in graph
        self.total_nodes = 0

        # Nodes in thin levels (starting from 0) and their mapping to their actual number {0: actual_node_number, 1: actual_node_number...}
        self.nodes_in_thin_levels_mapping = {}

        # Thin levels in graph
        self.thin_levels = []

        self.levels_of_nodes_in_thin = {}
        self.indegrees_of_nodes_in_thin = {}

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

        self.levels_of_nodes_in_thin, self.indegrees_of_nodes_in_thin = self.constructor.init_levels_of_nodes_in_thin()

        # A discrete action space. Nodes in thin levels will be chosen.
        # Action and observation spaces are determined according to matrix.
        self.action_space = gym.spaces.Discrete(len(self.nodes_in_thin_levels_mapping), start=list(self.levels_of_nodes_in_thin.keys())[0])

        # Observation space contains nodes' levels and indegrees which are in thin levels.
        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(len(self.levels_of_nodes_in_thin) + len(self.indegrees_of_nodes_in_thin),), dtype=np.int64)

        data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())
        observation = np.array(data)
        return observation, {}
    
    def render(self):
        self.graph.draw_graph()
    