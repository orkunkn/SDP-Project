from bisect import bisect_left

class Actions:
    
    def __init__(self, environment):
        self.env = environment


    def move_node_to_next_thin_level(self, node):

        if node not in self.env.G.nodes():
            return False

        # Sort thin_levels each time this method is called
        thin_levels_sorted = sorted(self.env.thin_levels)
        
        original_level = self.env.node_levels.get(node)
        if original_level is None:
            return False
        
        # Using bisect to find the correct insertion point and then adjust by one to move up
        pos = bisect_left(thin_levels_sorted, original_level)
        if pos == 0 or thin_levels_sorted[pos] != original_level:
            # Node level is already the highest or not found exactly in list (bisect_left might find insertion point)
            return False

        new_level = thin_levels_sorted[pos - 1]
        first_indegree = self.env.G.in_degree(node)
        
        while self.env.node_levels[node] != new_level:
            self.env.node_levels[node] -= 1 # Move node 1 level
            self.update_graph_after_movement(node, self.env.node_levels[node])
            
        indegree = self.env.G.in_degree(node)
        first_cost = max(0, 2 * first_indegree - 1)
        cost = max(0, 2 * indegree - 1)
        
        # Update the level cost and total cost
        self.env.level_costs[new_level] += cost
        self.env.level_costs[original_level] -= first_cost
        self.env.node_count_per_level[original_level] -= 1
        self.env.node_count_per_level[new_level] += 1

        self.env.indegree_dict = dict(self.env.G.in_degree())
        self.remove_empty_level(original_level)

        return True


    def move_node_to_next_level(self, node):
        if node not in self.env.G.nodes():
            return False
        
        original_level = self.env.node_levels.get(node)
        if original_level is None:
            return False
        
        self.env.node_levels[node] -= 1 # Move node 1 level
        indegree = self.env.G.in_degree(node)
        cost = max(0, 2 * indegree - 1)
        
        # Update the level cost and total cost
        self.env.level_costs[original_level - 1] += cost
        self.env.level_costs[original_level] -= cost
        self.env.node_count_per_level[original_level] -= 1
        self.env.node_count_per_level[original_level - 1] += 1

        self.update_graph_after_movement(node, original_level - 1)
        self.remove_empty_level(original_level)
        self.env.indegree_dict = dict(self.env.G.in_degree())
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
        # Remove the specified level and shift all higher levels down by one
        for i in range(level, max(self.env.level_costs.keys())):
            self.env.level_costs[i] = self.env.level_costs.get(i + 1)
            self.env.node_count_per_level[i] = self.env.node_count_per_level.get(i + 1)

        # Remove the last element that has now been duplicated
        self.env.level_costs.pop(max(self.env.level_costs.keys()), None)
        self.env.node_count_per_level.pop(max(self.env.node_count_per_level.keys()), None)


    def remove_empty_level(self, current_level):
        if current_level not in self.env.node_levels.values():
            self.remove_levels(current_level)
            # Adjust levels for all nodes above the current level in a single pass
            for node, level in self.env.node_levels.items():
                if level > current_level:
                    self.env.node_levels[node] -= 1
        else:
            return False

     