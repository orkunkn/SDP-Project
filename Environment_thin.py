import gymnasium as gym
import numpy as np
from graph import Graph
from constructor import Constructor
from actions import Actions
import networkx as nx
import time
import json
import os

class GraphEnv(gym.Env):
    def __init__(self, matrix):
        super(GraphEnv, self).__init__()
        self.k1 = 50
        self.k2 = 1.5
        self.k3 = 50
        self.h1 = 0.03
        self.h2 = 0.05
        self.h3 = 0.08
        self.time_step = 0
        self.total_move_node_time = 0
        self.total_calculate_metric_time = 0
        self.total_step_time = 0
        self.total_reward = 0
        self.check_count = 100
        self.matrix = matrix

        open("logfile.json", "w").close()

    def step(self, action):
        print(self.thin_levels)
       
        start_time = time.time()
        agent_choice = action
        # Check if the action is valid
        print("valid", self.valid_actions_mask)
        if action not in self.valid_actions_mask.keys():
            reward=self.total_reward
            terminated = False
            observation = self.current_observation()
            print("--------")
            print("obs", observation)
          
            print("--------")
            x={}
            print(observation, reward, terminated, False, x)
            return observation, reward, terminated, False, x

        # Continue with valid action processing
        if action in self.valid_actions_mask.keys() and action in self.nodes_in_thin_levels_mapping.keys():
            print("ww",self.nodes_in_thin_levels_mapping)
            print(agent_choice)
            node_to_move = self.nodes_in_thin_levels_mapping[agent_choice]
            old_node_level = self.levels[node_to_move]
            source_node_count = self.node_count_per_level.get(old_node_level, 0)

            move_node_start_time = time.time()
            self.actions.move_node_to_higher_level_thin(node_to_move)
            move_node_end_time = time.time()

            self.total_move_node_time += move_node_end_time - move_node_start_time

            is_node_moved = self.levels[node_to_move] != old_node_level
            if is_node_moved:
                self.constructor.calculate_graph_metrics()

            calculate_metric_end_time = time.time()
            self.total_calculate_metric_time += calculate_metric_end_time - move_node_end_time

            reward, info = self.calculate_reward(node_to_move, old_node_level, source_node_count)
            self.total_reward += reward

            print("total reward", self.total_reward)

            terminated = (len(self.thin_levels) <= 1) or (len(self.thin_levels) == 2 and max(self.levels.values()) == 1)

            observation = self.current_observation()

            end_time = time.time()
            self.total_step_time = end_time - start_time
            self.time_step += 1

            if self.time_step % self.check_count == 0:
                self.log_statistics()
            return observation, reward, terminated, False, info
           

        else:
            observation = self.current_observation()  # Get the current observation
            return observation, self.total_reward, False, False, {}


    def update_action_mask(self):
        self.valid_actions_mask = {}
        for action, node in self.nodes_in_thin_levels_mapping.items():
            parent_count = len(self.node_parents.get(node, []))
            grandparents_count = sum(len(self.node_parents.get(parent, [])) for parent in self.node_parents.get(node, []))
            if grandparents_count > parent_count:
                self.valid_actions_mask[action] = True
        print("updated space", self.valid_actions_mask)

    def current_observation(self):
        data = list(self.levels_of_nodes_in_thin.values()) + list(self.indegrees_of_nodes_in_thin.values())

        # Make sure the data is in the correct format (e.g., NumPy array) and matches the observation space definition
        observation = np.array(data, dtype=np.float32)  # Ensure the data type matches the observation space
        return observation


    def log_statistics(self):
        avg_move_node_time = self.total_move_node_time / self.check_count
        avg_calculate_metric_time = self.total_calculate_metric_time / self.check_count
        avg_reward = self.total_reward / self.check_count
        avg_step_time = self.total_step_time / self.check_count

        print("----", avg_reward)

        new_log_entry = {
            "time_step": self.time_step,
            "avg_reward": round(avg_reward, 2),
            "avg_move_node_time": round(avg_move_node_time, 6),
            "avg_calculate_metric_time": round(avg_calculate_metric_time, 6),
            "avg_step_time": round(avg_step_time, 6)
        }

        with open("logfile.json", "a") as f:
            json.dump(new_log_entry, f, indent=4)
            f.write("\n")

       
        self.total_move_node_time = 0
        self.total_calculate_metric_time = 0
        self.total_reward = 0



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
        info = {
            "ALC": round(self.ALC, 3),
            "ARL": round(self.ARL, 3),
            "AIR": round(self.AIR, 3)
        }
        return total_reward, info

    def reset(self, seed=None):
        self.G = nx.DiGraph()
        self.levels = {}
        self.node_parents = {}
        self.node_count_per_level = {}
        self.level_costs = {}
        self.total_nodes = 0
        self.nodes_in_thin_levels_mapping = {}
        self.thin_levels = []
        self.levels_of_nodes_in_thin = {}
        self.indegrees_of_nodes_in_thin = {}

        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

        self.graph = Graph(self)
        self.constructor = Constructor(self)
        self.actions = Actions(self, self.constructor)

        self.graph.convert_matrix_to_graph(self.matrix)
        self.constructor.calculate_graph_metrics()

        self.levels_of_nodes_in_thin, self.indegrees_of_nodes_in_thin = self.constructor.init_levels_of_nodes_in_thin()

        # Initialize the action space with the number of nodes in thin levels
        self.action_space = gym.spaces.Discrete(len(self.nodes_in_thin_levels_mapping))

        # Initialize the observation space based on the levels and indegrees of nodes in thin levels
        self.observation_space = gym.spaces.Box(low=0, high=1000000, shape=(len(self.levels_of_nodes_in_thin) + len(self.indegrees_of_nodes_in_thin),), dtype=np.int64)

        self.update_action_mask()  # Initialize the action mask based on the current state

        return self.current_observation(), {}

    def render(self):
        self.graph.draw_graph()

 