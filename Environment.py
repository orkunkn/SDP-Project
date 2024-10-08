import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx
import time
import json
import os
from torch.distributions import Distribution

# Used for preventing an error with MaskablePPO
Distribution.set_default_validate_args(False)

# Max level count of the matrices
MAX_ACTION_SPACE = 6765

MAX_OBS_SPACE = MAX_ACTION_SPACE * 3 + 6

class GraphEnv(gym.Env):

    def __init__(self, matrix, mode="train"):
        super(GraphEnv, self).__init__()

        # Reward coefficients
        self.k1 = 10
        self.h1 = 2.5
        self.k2 = 7.3
        self.k3 = 10
        self.h3 = 0.2
        self.k4 = 10
        self.h4 = 0.4

        # Used for logging
        self.check_count = 2048

        self.matrix = matrix
        self.mode = mode

        # A discrete action space. Nodes in thin levels will be chosen.
        # First action is level and second action is the action (move to next level or move to next thin level)
        # (If action < MAX_ACTION_SPACE --> thin move, else --> normal move)
        self.action_space = gym.spaces.Discrete(MAX_ACTION_SPACE * 2)

        # Observation space contains level details and average values in graph
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(MAX_OBS_SPACE,), dtype=np.float32)

        # Clear the log file at the start of each run
        open("logfile.json", "w").close()
        
    
    def valid_action_mask(self):
        action_masks = np.zeros(MAX_ACTION_SPACE * 2, dtype=bool)
        total_action = MAX_ACTION_SPACE * 2 - 1

        # Mask the action space for thin levels

        # Thin level move mask
        action_masks[self.thin_levels[1:]] = True
        # Normal move mask
        reverse_indices = total_action - self.thin_levels
        action_masks[reverse_indices] = True

        # if first level is thin level, mask it
        action_masks[total_action] = False

        return action_masks

    def step(self, action):
        
        # Decide the action
        move_action = "thin" if action < MAX_ACTION_SPACE else "normal"

        if move_action == "normal":
            action = MAX_ACTION_SPACE * 2 - 1 - action
        
        start_time = time.perf_counter()

        # Find the node in selected level which has lowest parent cost
        level_nodes = np.where(self.node_levels == action)[0]
        min_index = np.argmin(self.node_parents_cost_sum[level_nodes])
        node_to_move = level_nodes[min_index]       
        
        # Keep the old values for reward comparison
        old_node_level = self.node_levels[node_to_move]
        source_node_count = self.node_count_per_level[old_node_level]

        # Move the node to the next thin level
        if move_action == "thin":
            self.thin_move_count += 1
            self.actions.move_node_to_next_thin_level(node_to_move, source_node_count, old_node_level)

        # Move the node to the next level
        else:
            self.normal_move_count += 1
            self.actions.move_node_to_next_level(node_to_move, source_node_count, old_node_level)

        # Metrics are updated after move.
        self.constructor.calculate_graph_metrics()

        reward, info = self.calculate_reward(node_to_move, source_node_count)
        self.total_reward += reward

        # Learning is done if no thin levels are left or last thin level is the upmost level.
        thin_levels_len = len(self.thin_levels)
        if thin_levels_len <= 1:
            if thin_levels_len == 0 or self.thin_levels[0] == 0:
                self.terminated = True
        
        observation = self.create_observation()

        end_time = time.perf_counter()

        self.total_step_time += end_time - start_time
        self.time_step += 1

        # Log the current graph info
        if self.terminated or self.time_step % self.check_count == 0:
            self.log_info(info)

        # Returns observation, reward, terminated (is there any thin levels left), truncated (unnecessary so always False) and info.
        return observation, reward, self.terminated, False, info

    
    def calculate_reward(self, node, source_node_count):
        new_node_level = self.node_levels[node]
        node_moved_level = self.node_move_count[node]
        new_level_node_count = self.node_count_per_level[new_node_level]
        new_level_cost = self.level_costs[new_node_level]
        source_node_count -= 1

        # Source level node count reward
        part_1 = self.k1 * 10**(-source_node_count * self.h1 / self.ARL)

        # Node move count reward
        part_2 = - (self.k2 / 729) * (node_moved_level - 10)**3 if node_moved_level <= 20 else -10

        # Getting closer to ALC reward
        part_3 = self.k3 / (1 + self.h3 * (self.ALC - new_level_cost)) if new_level_cost < self.ALC \
                 else (self.k3 + 10) / (1 + 2 * self.h3 * (new_level_cost - self.ALC)) - 10
        
        # Getting closer to ARL reward
        part_4 = self.k4 / (1 + self.h4 * (self.ARL - new_level_node_count)) if new_level_node_count < self.ARL \
                 else (self.k4 + 10) / (1 + 2 * self.h4 * (new_level_node_count - self.ARL)) - 10
        
        # Used for logging
        self.part_1_total += part_1
        self.part_2_total += part_2
        self.part_3_total += part_3
        self.part_4_total += part_4

        # Normalize the reward between -1 and 1
        total_reward = (part_1 + part_2 + part_3 + part_4) / 40

        info = {"ALC": round(self.ALC, 1), "ARL": round(self.ARL, 1), "AIL": round(self.AIL, 1)}

        return total_reward, info

    
    # Used for logging info into a JSON file. Check count can be updated in init
    def log_info(self, info):

        avg_reward = self.total_reward / self.check_count
        avg_step_time = self.total_step_time / self.check_count
        avg_part_1 = self.part_1_total / self.check_count
        avg_part_2 = self.part_2_total / self.check_count
        avg_part_3 = self.part_3_total / self.check_count
        avg_part_4 = self.part_4_total / self.check_count

        new_log_entry = {
            "time_step": self.time_step,
            "avg_reward": round(avg_reward, 2),
            "part_1": round(avg_part_1, 2),
            "part_2": round(avg_part_2, 2),
            "part_3": round(avg_part_3, 2),
            "part_4": round(avg_part_4, 2),
            "info": info,
            "thin levels left": len(self.thin_levels),
            "avg_time": round(avg_step_time, 6),
            "thin/normal": round(self.thin_move_count/self.normal_move_count, 3)
        }

        # Log the info
        log_file_path = "logfile.json"
        log_data = []
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
           with open(log_file_path, "r") as f:
               log_data = json.load(f)

        log_data.append(new_log_entry)

        with open(log_file_path, "w") as f:
             json.dump(log_data, f, indent=4)

        # Reset metrics
        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0
        self.part_4_total = 0

    def create_observation(self):
        max_level_cost = np.max(self.level_costs)
        max_node_at_level = np.max(self.node_count_per_level)

        # Observation space arrays
        node_density = self.node_count_per_level / self.total_nodes
        move_norm = self.level_move_count / max_node_at_level
        level_cost_norm = self.level_costs / max_level_cost

        node_density_array = np.zeros(MAX_ACTION_SPACE, dtype=np.float32)
        move_norm_array = np.zeros(MAX_ACTION_SPACE, dtype=np.float32)
        level_cost_norm_array = np.zeros(MAX_ACTION_SPACE, dtype=np.float32)

        # Fill the gaps with 0 since obs space has to be fixed length
        node_density_array[:len(node_density)] = node_density
        move_norm_array[:len(move_norm)] = move_norm
        level_cost_norm_array[:len(level_cost_norm)] = level_cost_norm

        # Other metrics
        level_density = len(self.thin_levels) / self.level_count
        arl_norm = self.ARL / max_node_at_level
        alc_norm = self.ALC / max_level_cost
        ail_norm = self.AIL / np.max(self.level_indegrees)

        # Historical and contextual features
        recent_thin_level_changes = (self.first_thin_level_count - len(self.thin_levels)) / self.first_thin_level_count
        recent_move_ratio = self.thin_move_count / (self.normal_move_count)

        # Concatenate the values and normalize
        combined_features = np.concatenate([node_density_array, level_cost_norm_array, move_norm_array, [arl_norm, alc_norm, ail_norm, level_density, recent_thin_level_changes, recent_move_ratio]])
        normalized_features = combined_features / np.linalg.norm(combined_features, ord=2)

        observation = np.zeros(MAX_OBS_SPACE, dtype=np.float32)
        observation[:len(normalized_features)] = normalized_features

        return observation


    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None, options={}):

        self.G = nx.DiGraph()

        # A numpy array to keep node levels, index = node and value = level
        self.node_levels = []

        # A numpy array to keep node move counts, index = node and value = move count
        self.node_move_count = []

        # A numpy array to keep total move counts of nodes in the same level, used in obs space
        self.level_move_count = np.zeros(MAX_ACTION_SPACE, dtype=int)

        # A node counter for every level.
        self.node_count_per_level = []

        # A dictionary for cost of every level
        self.level_costs = []

        # A dictionary for indegree count of every level
        self.level_indegrees = []

        # A numpy array to keep node parents' costs' sum, used for finding the node in a level with the least parent cost sum
        self.node_parents_cost_sum = []

        # Thin levels in graph
        self.thin_levels = []

        # All levels in graph
        self.levels = []

        self.total_nodes = 0
        self.total_parents = 0
        self.total_cost = 0
        self.level_count = 0
        
        self.first_thin_level_count = -1

        # Updated in constructor
        self.AIL = 0
        self.ARL = 0
        self.ALC = 0
        self.AIR = 0

        # Used for logging
        self.terminated = False
        self.time_step = 0
        self.total_step_time = 0
        self.total_reward = 0
        self.part_1_total = 0
        self.part_2_total = 0
        self.part_3_total = 0
        self.part_4_total = 0

        # Start from 1 to prevent division by 0, used in obs space
        self.thin_move_count = 1
        self.normal_move_count = 1

        # Used for converting matrix to graph and drawing the graph
        self.graph = Graph(self)
        # Used by agent to required background calculations
        self.constructor = Constructor(self)
        # Used by agent for actions in graph
        self.actions = Actions(self)
    
        # Since reset is called before every learning process, these functions are done in reset
        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()
        self.first_thin_level_count = len(self.thin_levels)
        
        observation = self.create_observation()
        
        return observation, {}
    

    def render(self):
        self.graph.draw_graph()
