import numpy as np
class Actions:
    
    def __init__(self, environment):
        self.env = environment


    def move_node_to_next_thin_level(self, node):

        original_level = self.env.node_levels.get(node)
        temp_level = original_level

        index = np.where(self.env.thin_levels == original_level)[0][0]

        if index == 0:
            return False

        new_level = self.env.thin_levels[index - 1]
        first_indegree = self.env.G.in_degree(node)
        first_cost = max(0, 2 * first_indegree - 1)

        while temp_level != new_level:
            self.env.node_levels[node] -= 1 # Move node 1 level
            temp_level -= 1 # Used as new level in loop
            self.update_graph_after_movement(node, temp_level)
            
        indegree = self.env.G.in_degree(node)
        new_cost = max(0, 2 * indegree - 1)

        self.env.level_costs[original_level] -= first_cost
        self.env.level_costs[new_level] += new_cost
        self.env.node_count_per_level[original_level] -= 1
        self.env.node_count_per_level[new_level] += 1
        self.env.level_indegrees[original_level] -= first_indegree
        self.env.level_indegrees[new_level] += indegree
        self.env.node_move_count[node] += (original_level - new_level)

        self.remove_empty_level(original_level)

        return True


    def move_node_to_next_level(self, node):
        
        original_level = self.env.node_levels.get(node)
        if original_level == 0:
            return False
        
        self.env.node_levels[node] -= 1 # Move node 1 level

        first_indegree = self.env.G.in_degree(node)
        first_cost = max(0, 2 * first_indegree - 1)

        self.update_graph_after_movement(node, original_level - 1)
            
        indegree = self.env.G.in_degree(node)
        new_cost = max(0, 2 * indegree - 1)
        
        self.env.level_costs[original_level] -= first_cost
        self.env.level_costs[original_level - 1] += new_cost
        self.env.node_count_per_level[original_level] -= 1
        self.env.node_count_per_level[original_level - 1] += 1
        self.env.level_indegrees[original_level] -= first_indegree
        self.env.level_indegrees[original_level - 1] += indegree
        self.env.node_move_count[node] += 1

        self.remove_empty_level(original_level)
        return True


    def update_graph_after_movement(self, node, new_level):
        # Filter parents that are now on the same level and collect grandparents if needed.
        parents_same_level = []
        grandparents_to_connect = []

        for parent in list(self.env.G.predecessors(node)):
            if self.env.node_levels[parent] == new_level:
                parents_same_level.append(parent)
                grandparents_to_connect.extend(self.env.G.predecessors(parent))

        # Remove edges from parents on the same level
        for parent in parents_same_level:
            self.env.G.remove_edge(parent, node)

        # Add edges from grandparents to node, avoiding duplicate edges
        for grandparent in set(grandparents_to_connect):
            if not self.env.G.has_edge(grandparent, node):
                self.env.G.add_edge(grandparent, node)


    def remove_levels(self, level):
        # Determine the maximum level
        max_level = max(self.env.level_costs.keys())

        # Remove the specified level and shift all higher levels down by one
        for i in range(level, max_level):
            self.env.level_costs[i] = self.env.level_costs.get(i + 1)
            self.env.node_count_per_level[i] = self.env.node_count_per_level.get(i + 1)
            self.env.level_indegrees[i] = self.env.level_indegrees.get(i + 1)

        # Handle thin_levels separately if the condition is met
        if len(self.env.thin_levels) < self.env.first_thin_level_count * 0.1:
            self.env.thin_levels = self.env.thin_levels[self.env.thin_levels != level]
            self.env.thin_levels[self.env.thin_levels > level] -= 1

        # Remove the now redundant last entries from dictionaries
        self.env.level_costs.pop(max_level, None)
        self.env.node_count_per_level.pop(max_level, None)
        self.env.level_indegrees.pop(max_level, None)


    def remove_empty_level(self, current_level):
        if current_level not in self.env.node_levels.values():
            self.remove_levels(current_level)
            # Adjust levels for all nodes above the current level in a single pass
            for node, level in self.env.node_levels.items():
                if level > current_level:
                    self.env.node_levels[node] -= 1
        else:
            return False

     