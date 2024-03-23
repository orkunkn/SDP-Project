
class Actions:
    
    def __init__(self, environment, constructor):
        self.env = environment
        self.con = constructor


    def move_node_to_higher_level_thick(self, node, levels_to_drop):
        new_level = self.env.levels[node] - levels_to_drop
        if node not in self.env.G.nodes() or new_level < 0:
            return

        original_level = self.env.levels[node]

        self.env.levels[node] = new_level # Move node
        self.con.update_graph_after_movement(node, new_level)
        self.remove_empty_level(original_level)


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
            self.con.update_graph_after_movement(node, self.env.levels[node])

        self.remove_empty_level(original_level)


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
     