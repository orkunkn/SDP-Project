import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx
import time
# wallclock time or cpu tick (updated)
# logları her time stepde bir yap
# optimization
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
        self.time_step = 0
        self.total_step_time = 0
        self.total_reward = 0
        self.check_count = 100
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0
        self.matrix = matrix

        # Clear the log file at the start of each run
        open("logfile.json", "w").close()
        
        
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

        start_time = time.perf_counter()

        # Keep the old values for reward comparison
        old_node_level = self.levels[node_to_move]
        source_node_count = self.node_count_per_level[old_node_level]

        # Move the node one level upper
        self.actions.move_node_to_higher_level_thin(node_to_move)

        is_node_moved = self.levels[node_to_move] != old_node_level
        if is_node_moved:
            # Metrics are updated after movement.
            self.constructor.calculate_graph_metrics()

        reward, info = self.calculate_reward(node_to_move, old_node_level, source_node_count)
        self.total_reward += reward

        # Learning is done if there is one or zero thin level left.
        terminated = (len(self.thin_levels) <= 1) or (len(self.thin_levels) == 2 and max(self.levels.values()) == 1)

        # Convert object to a list
        data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())
        
        # Convert list to an array
        observation = np.array(data)

        end_time = time.perf_counter()
        self.total_step_time += end_time - start_time

        self.time_step += 1

        if self.time_step % self.check_count == 0:
            self.log_info(info)
            self.total_step_time = 0
            self.total_reward = 0
            self.part_1_total = 0
            self.part_2_total = 0
            self.part_3_total = 0

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

        # info = {"part 1":part_1, "part 2":part_2, "part 3":part_3, "ALC":self.ALC, "ARL":self.ARL, "done": len(self.thin_levels) <= 1}

        info = {
            "ALC": round(self.ALC, 3),
            "ARL": round(self.ARL, 3),
            "AIR": round(self.AIR, 3)
        }
        return total_reward, info
    
    def log_info(self, info):

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
            "thin levels left": len(self.thin_levels),
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
        self.actions = Actions(self)
    
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
    