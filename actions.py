
class Actions:
    
    def __init__(self, environment):
        self.env=environment


    def move_node_to_higher_level(self, node):
        if node not in self.G.nodes():
            print(f"Node {node} does not exist in the graph.")
            return

        original_level = self.levels[node]
        moved = False
        current_air = self.AIR

        for new_level in range(original_level - 1, -1, -1):
           
            grandparents_count = self.calculate_total_grandparents(node)
          
            if grandparents_count > current_air:
                self.move_node(node, new_level)
                self.update_graph_after_movement(node, new_level)
                self.calculate_graph_metrics()  
                moved = True
                print(f"Node {node} moved to level {new_level}.")
                break
            else:
                
                self.G=self.create_graph_from_indegree(self.init_parents,self.init_levels)
                print(f"Node {node} not moved, grandparents count ({grandparents_count}) is not greater than AIR ({current_air}).")
                break
               

        if moved:
            x=self.remove_empty_level(original_level)
            print("Node moved successfully.", node)
        else:
            print(f"Node {node} did not improve metrics. It remains at level {original_level}.")

        
    def move_node(self, node, new_level):
        self.levels[node] = new_level


    def remove_levels(self, level):
        keys_to_remove = [key for key, val in self.levels.items() if val == level]
    
        for key in keys_to_remove:
            del self.levels[key]


    def remove_empty_level(self, current_level):
        all_level = self.levels.values()
        print(current_level not in all_level)
        if current_level not in all_level:
            self.remove_levels(current_level)
            for node,level in self.levels.items():
                if level > current_level:
                    self.levels[node]-=1
        else:
            return False
     
    def move_nodes_from_small_levels(self):
        level_counts = {level: 0 for level in set(self.levels.values())}
        for node in self.G.nodes():
            level_counts[self.levels[node]] += 1

        for level, count in level_counts.items():
            if count in [1, 2, 3]:
                # Move all nodes from this level to the next higher level
                for node in [n for n, lvl in self.levels.items() if lvl == level]:
                    new_level = level - 1
                    self.move_node(node, new_level)
                    self.update_graph_after_movement(node, new_level)
                self.calculate_graph_metrics()
                self.remove_empty_level(level)
                print(f"Nodes from level {level} moved to level {new_level}.")

    

