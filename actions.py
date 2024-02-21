
class Actions:
    
    def __init__(self, environment, constructor):
        self.env = environment
        self.con = constructor


    def move_node_to_higher_level(self, node, levels_to_drop):
        new_level = self.env.levels[node] - levels_to_drop
        if node not in self.env.G.nodes() or new_level < 0:
            return

        original_level = self.env.levels[node]

        self.move_node(node, new_level)
        self.con.update_graph_after_movement(node, new_level)
        self.remove_empty_level(original_level)

        
    def move_node(self, node, new_level):
        self.env.levels[node] = new_level


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
     