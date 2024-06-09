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

        # Determine the update function based on the mode
        update_graph = self.update_graph_after_movement_with_weights if self.env.mode == "run" else self.update_graph_after_movement_without_weights

        # Continue updating the graph until the node reaches the target level
        while temp_level != target_level:
            index = np.where(self.env.levels == temp_level)[0][0]
            new_level = self.env.levels[index - 1]
            temp_level = new_level  # Update the current level for the next iteration
            update_graph(node, temp_level)

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

        if self.env.mode == "run":
            self.update_graph_after_movement_with_weights(node, target_level)
        else:
            self.update_graph_after_movement_without_weights(node, target_level)

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


    def update_graph_after_movement_with_weights(self, node, new_level):
        # Filter parents that are now on the same level
        node_predecessors = set(self.env.G.predecessors(node))
        parents_same_level = [parent for parent in node_predecessors if self.env.node_levels[parent] == new_level]

        # Collect grandparents and edges to add
        edges_to_add = set()
        for parent in parents_same_level:
            grandparents = set(self.env.G.predecessors(parent))
            for grandparent in grandparents:
                if grandparent not in node_predecessors:
                    weight = self.env.G[grandparent][parent]['weight']
                    edges_to_add.add((grandparent, node, weight))

        # Remove edges from parents on the same level
        self.env.G.remove_edges_from([(parent, node) for parent in parents_same_level])
        
        # Add edges from grandparents to node
        self.env.G.add_weighted_edges_from(edges_to_add)

    def update_graph_after_movement_without_weights(self, node, new_level):
        # Filter parents that are now on the same level
        node_predecessors = set(self.env.G.predecessors(node))
        parents_same_level = [parent for parent in node_predecessors if self.env.node_levels[parent] == new_level]

        # Collect grandparents and edges to add
        edges_to_add = set()
        for parent in parents_same_level:
            grandparents = set(self.env.G.predecessors(parent))
            for grandparent in grandparents:
                if grandparent not in node_predecessors:
                    edges_to_add.add((grandparent, node))

        # Remove edges from parents on the same level
        self.env.G.remove_edges_from([(parent, node) for parent in parents_same_level])
        
        # Add edges from grandparents to node
        self.env.G.add_edges_from(edges_to_add)
