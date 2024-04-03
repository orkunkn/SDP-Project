import gymnasium as gym
import numpy as np
from graph import Graph
from constructor_thick import Constructor
from actions import Actions
import networkx as nx
import time
# logları her time stepde bir yap
# reward part 2
import json
import os

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
        self.check_count = 2000
        self.time_step = 0
        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0

        # Clear the log file at the start of each run
        open("logfile.json", "w").close()
        
        
    def step(self, action):

        # Node is chosen by agent.
        node_to_move = action[0]
        levels_to_drop = action[1]
        terminated = False
        
        start_time = time.perf_counter()

        # Keep the old values for reward comparison
        old_node_level = self.levels[node_to_move]
        source_node_count = self.node_count_per_level[old_node_level]

        # Move the node one level upper
        is_node_moved = self.actions.move_node_to_higher_level_thick(node_to_move, levels_to_drop)
        
        if is_node_moved:
            # Metrics are updated after movement.
            self.constructor.calculate_graph_metrics()

        reward, info = self.calculate_reward(node_to_move, old_node_level, source_node_count)
        self.total_reward += reward

        data = list(self.levels.values()) + list(self.indegree_dict.values())
        observation = np.array(data)

        end_time = time.perf_counter()
        self.total_step_time += end_time - start_time

        self.time_step += 1

        self.log_info(info)
        sa = all(level_cost > 0.8 * self.first_ALC for level_cost in self.level_costs.values())
        terminated = sa or max(self.levels.values()) == 0
        if sa:
            print("saaa")

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
            self.part_1_total += part_1
            self.part_2_total += part_2
            self.part_3_total += part_3
            total_reward = part_1 + part_2 + part_3


        info = {
            "ALC": round(self.ALC, 3),
            "ARL": round(self.ARL, 3),
            "AIR": round(self.AIR, 3)
        }
        return total_reward, info
    
    def log_info(self, info):

        if self.time_step % self.check_count != 0:
            return
        
        avg_reward = self.total_reward / self.check_count
        avg_step_time = self.total_step_time / self.check_count
        avg_part_1 = self.part_1_total / self.check_count
        avg_part_2 = self.part_2_total / self.check_count
        avg_part_3 = self.part_3_total / self.check_count

        new_log_entry = {
            "time_step": self.time_step,
            "avg_reward": round(avg_reward, 2),
            "part_1": round(avg_part_1, 2),
            "part_2": round(avg_part_2, 2),
            "part_3": round(avg_part_3, 2),
            "info": info,
            "avg_time": round(avg_step_time, 6)
        }

        log_file_path = "logfile.json"
        log_data = []
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
            with open(log_file_path, "r") as f:
                log_data = json.load(f)

        log_data.append(new_log_entry)

        with open(log_file_path, "w") as f:
            json.dump(log_data, f, indent=4)

        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0
    

    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None):

        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level.
        self.levels = {}

        # A dictionary mapping each node to its parent node.
        self.node_parents = {}

        # A node counter for every level.
        self.node_count_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}
        
        self.indegree_dict = {}

        # Number of total nodes in graph
        self.total_nodes = 0

        # Updated in constructor
        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

        # Used for converting matrix to graph and drawing the graph
        self.graph = Graph(self)
        # Used by agent to required background calculations
        self.constructor = Constructor(self)
        # Used by agent for actions in graph
        self.actions = Actions(self)
    
        # Since reset is called before every learning process, these functions are done in reset
        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()

        self.first_ALC = self.ALC
        # Action and observation spaces are determined according to matrix.
        self.action_space = gym.spaces.MultiDiscrete([self.total_nodes, 10])

        # Observation space contains nodes' levels and indegrees.
        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(len(self.levels) + len(self.indegree_dict),), dtype=np.int64)

        data = list(self.levels.values()) + list(self.indegree_dict.values())
        observation = np.array(data)
        return observation, {}
    
    def render(self):
        self.graph.draw_graph()
    