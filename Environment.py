import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx
import time
import json
import os

MAX_ACTION_SPACE = 22616
MAX_OBS_SPACE = MAX_ACTION_SPACE * 2 + 1
# PR02R node count in thin levels = 22616
# (ignoring pwtk, Si41Ge41H72, af_0_k101, ohne2)

class GraphEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphEnv, self).__init__()

        # Reward coefficients
        self.k1 = 50
        self.k2 = 3.5
        self.k3 = 50
        self.h1 = 0.03
        self.h3 = 0.08

        # Used for logging
        self.check_count = 1000

        self.matrix = matrix

        # A multi discrete action space. Nodes in thin levels will be chosen.
        # First action is node and second action is the action (move to next level or move to next thin level)
        self.action_space = gym.spaces.MultiDiscrete([MAX_ACTION_SPACE + 1, 2])

        # Observation space contains nodes' levels and indegrees which are in thin levels and how many nodes left to move.
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(MAX_OBS_SPACE,), dtype=np.float32)

        # Clear the log file at the start of each run
        open("logfile.json", "w").close()
        
        
    def step(self, action):

        # Node and action are chosen by agent.
        agent_choice_node = action[0]
        agent_choice_action = action[1]

        # Finding the node from "nodes in thin level" dict
        if agent_choice_node in self.nodes_in_thin_levels_mapping:
            node_to_move = self.nodes_in_thin_levels_mapping[agent_choice_node]
        
        # If the chosen node does not exist or not in a thin level
        else:
            reward = -100

            terminated = False

            observation = self.current_observation

            # Returns observation, reward, terminated (always False in this case), truncated (unnecessary so always False) and info.
            return observation, reward, terminated, False, {}

        start_time = time.perf_counter()

        # Keep the old values for reward comparison
        old_node_level = self.node_levels[node_to_move]
        source_node_count = self.node_count_per_level[old_node_level]
        
        # Move the node to the next thin level
        if agent_choice_action == 0:
            is_node_moved = self.actions.move_node_to_next_thin_level(node_to_move)

        # Move the node to the next level
        else:
            is_node_moved = self.actions.move_node_to_next_level(node_to_move)

        # Metrics are updated if node is moved to another level.
        if is_node_moved:
            self.constructor.calculate_graph_metrics()

        reward, info = self.calculate_reward(node_to_move, old_node_level, source_node_count)
        self.total_reward += reward

        # Learning is done if no thin levels are left.
        terminated = len(self.thin_levels) == 0

        observation = self.create_observation()

        end_time = time.perf_counter()
        self.total_step_time += end_time - start_time
        self.time_step += 1

        self.log_info(info)

        # Returns observation, reward, terminated (is there any thin levels left), truncated (unnecessary so always False) and info.
        return observation, reward, terminated, False, info
        

    def calculate_reward(self, node, old_node_level, source_node_count):
        new_level_cost = self.level_costs[self.node_levels[node]]
        part_1, part_2, part_3 = 0, 0, 0

        # Node is not moved
        if old_node_level == self.node_levels[node]:
            total_reward = -50
        else:
            # Reward for getting closer to deleting a level
            part_1 = self.S * self.k1 * 10**(-source_node_count * self.h1)

            # How long did node move (further move --> bigger penalty)
            part_2 = (-1/self.S) * self.k2 * (old_node_level - self.initial_node_levels[node])

            # Level cost comparison
            if new_level_cost < self.ALC:
                part_3 = self.S * self.k3 / (1 + self.h3 * abs(new_level_cost - self.ALC))
            else:
                part_3 = self.S * self.k3 / (1 + self.h3 * 3 * abs(new_level_cost - self.ALC))

            # Used for logging
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
    
    # Used for logging info into a JSON file. Check count can be updated in init
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

        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0


    def create_observation(self):
        # Combine the levels and indegrees of nodes from the graph into a single list
        data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())
        
        # Convert the combined data into a numpy array
        observation_data = np.array(data, dtype=np.float32)
        
        # Increase each element by the number of levels to adjust the scale
        observation_data = np.append(observation_data, max(self.nodes_in_thin_levels_mapping.keys()))
        
        # Normalizing the array to have values between 0 and 1
        scale_factor = 1 / (MAX_ACTION_SPACE - 1)
        normalized_obs = scale_factor * observation_data

        # Initialize a padded array with zeros to match the maximum observation space size
        padded_observation = np.zeros(MAX_OBS_SPACE, dtype=np.float32)
        
        # Copy the observation data into the padded array
        padded_observation[:len(normalized_obs)] = normalized_obs

        self.current_observation = padded_observation

        # Return the padded observation array
        return self.current_observation

    
    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None):

        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level {node: level}
        self.node_levels = {}

        # Used to see how far did a node moved from its initial level
        self.initial_node_levels = {}

        # A node counter for every level.
        self.node_count_per_level = {}

        # An indegree dictionary for every node
        self.indegree_dict = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        # Nodes in thin levels (starting from 0) and their mapping to their actual number {0: actual_node_number, 1: actual_node_number...}
        self.nodes_in_thin_levels_mapping = {}

        # Thin levels in graph
        self.thin_levels = set()

        self.levels_of_nodes_in_thin = {}
        self.indegrees_of_nodes_in_thin = {}

        # Updated in constructor
        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

        # Used for logging
        self.time_step = 0
        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0

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

        observation = self.create_observation()

        self.S = len(self.levels_of_nodes_in_thin) / MAX_ACTION_SPACE
        
        return observation, {}
    

    def render(self):
        self.graph.draw_graph()
    