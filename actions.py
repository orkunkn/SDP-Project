import numpy as np
class Actions:
    
    def __init__(self, environment):
        self.env = environment


    def move_node_to_next_thin_level(self, node, source_node_count, original_level):
        thin_index = np.where(self.env.thin_levels == original_level)[0][0]
        target_level = self.env.thin_levels[thin_index - 1]

        first_indegree = self.env.G.in_degree(node)
        first_cost = max(0, 2 * first_indegree - 1)

        # Find the parent nodes (predecessors)
        parent_nodes = list(self.env.G.predecessors(node))

        # Calculate the sum of in-degrees of the parent nodes
        first_parent_indegree_sum = sum(self.env.G.in_degree(n) for n in parent_nodes)

        temp_level = original_level

        while temp_level != target_level:
            index = np.where(self.env.levels == temp_level)[0][0]
            new_level = self.env.levels[index - 1]
            temp_level = new_level  # Used as new level in loop
            self.update_graph_after_movement(node, temp_level)

        self.env.node_levels[node] = new_level

        self.finalize_movement(node, original_level, target_level, first_indegree, first_cost, first_parent_indegree_sum)

        if source_node_count == 1:
            self.env.levels = np.delete(self.env.levels, np.where(self.env.levels == original_level)[0])
            self.env.level_count -= 1

    def move_node_to_next_level(self, node, source_node_count, original_level):
        index = np.where(self.env.levels == original_level)[0][0]
        target_level = self.env.levels[index - 1]

        self.env.node_levels[node] = target_level  # Move node 1 level

        first_indegree = self.env.G.in_degree(node)
        first_cost = max(0, 2 * first_indegree - 1)

        # Find the parent nodes (predecessors)
        parent_nodes = list(self.env.G.predecessors(node))

        # Calculate the sum of in-degrees of the parent nodes
        first_parent_indegree_sum = sum(2 * self.env.G.in_degree(n) - 1 for n in parent_nodes)

        self.update_graph_after_movement(node, target_level)

        self.finalize_movement(node, original_level, target_level, first_indegree, first_cost, first_parent_indegree_sum)

        if source_node_count == 1:
            self.env.levels = np.delete(self.env.levels, np.where(self.env.levels == original_level)[0])
            self.env.level_count -= 1

    def finalize_movement(self, node, original_level, target_level, first_indegree, first_cost, first_parent_indegree_sum):
        indegree = self.env.G.in_degree(node)
        new_cost = max(0, 2 * indegree - 1)

        self.env.level_costs[original_level] -= first_cost
        self.env.level_costs[target_level] += new_cost
        self.env.node_count_per_level[original_level] -= 1
        self.env.node_count_per_level[target_level] += 1
        self.env.level_indegrees[original_level] -= first_indegree
        self.env.level_indegrees[target_level] += indegree
        self.env.node_move_count[node] += (original_level - target_level)
        self.env.level_move_count[target_level] += (original_level - target_level)
        self.env.total_parents += (indegree - first_indegree)
        self.env.total_cost += (new_cost - first_cost)
        self.env.node_parents_cost_sum[node] -= first_parent_indegree_sum
        # Find the parent nodes (predecessors)
        parent_nodes = list(self.env.G.predecessors(node))

        # Calculate the sum of in-degrees of the parent nodes
        parent_indegree_sum = sum(2 * self.env.G.in_degree(n) - 1 for n in parent_nodes)
        self.env.node_parents_cost_sum[node] += parent_indegree_sum


    def update_graph_after_movement(self, node, new_level):
        # Filter parents that are now on the same level and collect grandparents if needed.
        parents_same_level = []
        grandparents_map = {}  # To store grandparents for each parent

        for parent in list(self.env.G.predecessors(node)):
            if self.env.node_levels[parent] == new_level:
                parents_same_level.append(parent)
                # Collect grandparents for each parent
                grandparents_map[parent] = list(self.env.G.predecessors(parent))

        # Remove edges from parents on the same level
        for parent in parents_same_level:
            self.env.G.remove_edge(parent, node)

        # Add edges from grandparents to node, avoiding duplicate edges
        for parent in parents_same_level:
            for grandparent in set(grandparents_map[parent]):  # Ensure uniqueness
                if not self.env.G.has_edge(grandparent, node):
                    weight = self.env.G[grandparent][parent]['weight']
                    self.env.G.add_edge(grandparent, node, weight=weight)
