import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx
import time
import json
import os

MAX_ACTION_SPACE = 88
MAX_OBS_SPACE = 266 + 3

# Used for normalizing obs space
SCALE_NUM = 10000

class GraphEnv(gym.Env):

    def __init__(self, matrix):
        super(GraphEnv, self).__init__()

        # Reward coefficients
        self.k1 = 50
        self.k2 = 1
        self.k3 = 50
        self.h1 = 0.03
        self.h3 = 0.08

        # Used for logging
        self.check_count = 512

        self.matrix = matrix

        # A multi discrete action space. Nodes in thin levels will be chosen.
        # First action is level and second action is the action (move to next level or move to next thin level)
        # self.action_space = gym.spaces.MultiDiscrete([MAX_ACTION_SPACE, 2])
        self.action_space = gym.spaces.Discrete(MAX_ACTION_SPACE * 2)

        # Observation space contains thin levels, level details and average values in graph
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(MAX_OBS_SPACE,), dtype=np.float32)

        # Clear the log file at the start of each run
        open("logfile.json", "w").close()
        
    
    def valid_action_mask(self):
        # Initialize the action mask array
        action_masks = np.zeros(MAX_ACTION_SPACE * 2, dtype=int)

        # Set valid masks for the first action based on thin_levels
        for i in self.thin_levels:
            action_masks[i] = 1
            if self.thin_levels.index(i) == 0:
                action_masks[i] = 0
            reverse_index = MAX_ACTION_SPACE * 2 - 1 - i
            action_masks[reverse_index] = 1

        return action_masks

    def step(self, action):

        # Node and action are chosen by agent.
        agent_choice_level = action

        # Move the node to the next thin level
        if agent_choice_level < MAX_ACTION_SPACE:
            move_action = "thin"

        # Move the node to the next level
        else:
            move_action = "normal"
        
        start_time = time.perf_counter()
        if agent_choice_level > max(self.node_levels.values()):
            agent_choice_level = MAX_ACTION_SPACE * 2 - 1 - agent_choice_level
            
        for node, level in self.node_levels.items():
            if level == agent_choice_level:
                node_to_move = node

        # Keep the old values for reward comparison
        old_node_level = self.node_levels.get(node_to_move)
        source_node_count = self.node_count_per_level.get(old_node_level)
        
        # Move the node to the next thin level
        if move_action == "thin":
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
        # terminated = len(self.thin_levels) == 0 or (len(self.thin_levels) == 1 and self.thin_levels_mapping.get(0) == 0)
        terminated = len(self.thin_levels) == 0 or (len(self.thin_levels) == 1 and self.thin_levels[0] == 0)

        observation = self.create_observation()

        end_time = time.perf_counter()
        self.total_step_time += end_time - start_time
        self.time_step += 1
        
        if terminated:
            self.log_info(info)
            # self.render()
        
        # Returns observation, reward, terminated (is there any thin levels left), truncated (unnecessary so always False) and info.
        return observation, reward, terminated, False, info


    def calculate_reward(self, node, old_node_level, source_node_count):
        
        new_node_level = self.node_levels.get(node)
        new_level_cost = self.level_costs[new_node_level]
        part_1, part_2, part_3 = 0, 0, 0

        # Node is not moved
        if old_node_level == new_node_level:
            total_reward = -50
        else:
            # Reward for getting closer to deleting a level
            part_1 = self.S * self.k1 * 10**(-source_node_count * self.h1)

            # How long did node move (further move --> bigger penalty)
            part_2 = -self.S * self.k2 * (self.initial_node_levels.get(node) - new_node_level)

            # Level cost comparison
            if new_level_cost < self.ALC:
                part_3 = self.S * self.k3 / (1 + self.h3 * (self.ALC - new_level_cost))
            else:
                part_3 = self.S * self.k3 / (1 + self.h3 * 3 * (new_level_cost - self.ALC))

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
        
        #if self.time_step % self.check_count != 0:
        #    return
        
        avg_reward = self.total_reward / self.time_step
        avg_step_time = self.total_step_time / self.time_step
        avg_part_1 = self.part_1_total / self.time_step
        avg_part_2 = self.part_2_total / self.time_step
        avg_part_3 = self.part_3_total / self.time_step
        self.avg_move = self.avg_move / self.time_step

        new_log_entry = {
            "time_step": self.time_step,
            "avg_reward": round(avg_reward, 2),
            "part_1": round(avg_part_1, 2),
            "part_2": round(avg_part_2, 2),
            "part_3": round(avg_part_3, 2),
            "info": info,
            "avg move": self.avg_move,
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
        self.avg_move = 0


    def create_observation(self):

        padded_levels = [0] * MAX_ACTION_SPACE
        
        # Copy the observation data into the padded array
        padded_levels[:len(self.thin_levels)] = self.thin_levels

        # Combine the levels and indegrees of nodes from the graph into a single list
        data = padded_levels + list(self.node_count_per_level.values()) + list(self.level_costs.values())
        """
        print("action:", len(self.thin_levels))
        print("max level:", max(self.node_levels.values()))
        print("obs:", len(self.thin_levels + list(self.node_count_per_level.values()) + list(self.level_costs.values())))
        """

        # Convert the combined data into a numpy array
        observation_data = np.array(data, dtype=np.float32)
        
        # Increase each element by the number of levels to adjust the scale
        observation_data = np.append(observation_data, [self.AIR, self.ALC, self.ARL])
        
        # Normalizing the array to have values between 0 and 1
        scale_factor = 1 / SCALE_NUM
        normalized_obs = scale_factor * observation_data

        # Initialize a padded array with zeros to match the maximum observation space size
        padded_observation = np.zeros(MAX_OBS_SPACE, dtype=np.float32)
        
        # Copy the observation data into the padded array
        padded_observation[:len(normalized_obs)] = normalized_obs

        self.current_observation = padded_observation

        # Return the padded observation array
        return self.current_observation

    
    # Used for reseting the environment. Do not change function inputs or return statement
    def reset(self, seed=None, options={}):

        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level {node: level}
        self.node_levels = {}

        # Used to see how far did a node moved from its initial level
        self.initial_node_levels = {}

        # A node counter for every level.
        self.node_count_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        # Thin levels in graph
        self.thin_levels = []

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
        self.avg_move = 0

        # Used for converting matrix to graph and drawing the graph
        self.graph = Graph(self)
        # Used by agent to required background calculations
        self.constructor = Constructor(self)
        # Used by agent for actions in graph
        self.actions = Actions(self)
    
        # Since reset is called before every learning process, these functions are done in reset
        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()

        observation = self.create_observation()

        self.S = max(self.node_levels.values()) / MAX_ACTION_SPACE 

        return observation, {}
    

    def render(self):
        self.graph.draw_graph()
