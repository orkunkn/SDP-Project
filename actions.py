
class Actions:
    
    def __init__(self, environment):
        self.env = environment


    def move_node_to_higher_level_thick(self, node, levels_to_drop):
        original_level = self.env.levels[node]
        new_level = original_level - levels_to_drop

        if node not in self.env.G.nodes() or new_level < 0 or levels_to_drop == 0:
            return False
        
        for i in range(1, levels_to_drop + 1):
            self.env.levels[node] -= 1  # Move node 1 level
            self.update_graph_after_movement(node, original_level - i)

        self.remove_empty_level(original_level)
        return True

    def move_node_to_higher_level_thin(self, node):

        if node not in self.env.G.nodes():
            return
        
        # Find the index of the provided number in the array
        index = self.env.thin_levels.index(self.env.levels[node])

        # Return the value before the provided number
        if index == 0:
            return
        else:
            new_level = self.env.thin_levels[index - 1]

        original_level = self.env.levels[node]

        while self.env.levels[node] != new_level:
            self.env.levels[node] -= 1 # Move node 1 level
            self.update_graph_after_movement(node, self.env.levels[node])

        self.remove_empty_level(original_level)

    def update_graph_after_movement(self, node, new_level):
        # Remove edges from parents that are now on the same level
        for parent in list(self.env.G.predecessors(node)):
            if self.env.levels[parent] == new_level:
                self.env.G.remove_edge(parent, node)
                # Add edges from grandparents, if any
                for grandparent in self.env.G.predecessors(parent):
                    self.env.G.add_edge(grandparent, node)

    def remove_levels(self, level):
        keys_to_remove = [key for key, val in self.env.levels.items() if val == level]
    
        for key in keys_to_remove:
            del self.env.levels[key]


    def remove_empty_level(self, current_level):
        all_level = self.env.levels.values()
        if current_level not in all_level:
            self.remove_levels(current_level)
            for node,level in self.env.levels.items():
                if level > current_level:
                    self.env.levels[node]-=1
        else:
            return False
     