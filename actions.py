
class Actions:
    
    def __init__(self, environment, constructor):
        self.env=environment
        self.con = constructor


    def move_node_to_higher_level(self, node, new_level):
        if node not in self.env.G.nodes() or self.env.levels[node] - new_level < 0:
            return

        original_level = self.env.levels[node]

        self.move_node(node, new_level)
        self.con.update_graph_after_movement(node, self.env.levels[node] - new_level)
        self.remove_empty_level(original_level)

        
    def move_node(self, node, new_level):
        self.env.levels[node] = self.env.levels[node] - new_level


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
     
    def move_nodes_from_small_levels(self):
        level_counts = {level: 0 for level in set(self.env.levels.values())}
        for node in self.env.G.nodes():
            level_counts[self.env.levels[node]] += 1

        for level, count in level_counts.items():
            if count in [1, 2, 3]:
                # Move all nodes from this level to the next higher level
                for node in [n for n, lvl in self.env.levels.items() if lvl == level]:
                    new_level = level - 1
                    self.move_node(node, new_level)
                    self.con.update_graph_after_movement(node, new_level)
                self.con.calculate_graph_metrics()
                self.remove_empty_level(level)
                print(f"Nodes from level {level} moved to level {new_level}.")

    

