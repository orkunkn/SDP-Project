
class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):
        self.env.total_nodes = self.env.G.number_of_nodes()
        self.env.indegree_dict = {node: self.env.G.in_degree(node) for node in self.env.G.nodes()}
        
        self.env.level_costs = {}
        self.env.node_count_per_level = {}
        total_cost = 0
        
        for node, level in self.env.levels.items():
            indegree = self.env.indegree_dict[node]
            cost = max(0, 2 * indegree - 1)

            if level in self.env.level_costs:
                self.env.level_costs[level] += cost
            else:
                self.env.level_costs[level] = cost
            total_cost += cost

            # Count nodes in every level
            if level in self.env.node_count_per_level:
                self.env.node_count_per_level[level] += 1
            else:
                self.env.node_count_per_level[level] = 1
            
        max_level = max(self.env.levels.values()) + 1
        total_parents = sum(self.env.indegree_dict.values())

        # 3 criterias to control
        # Average Indegree
        self.env.AIR = total_parents / self.env.total_nodes if self.env.total_nodes else 0
        # Average Row per Level
        self.env.ARL = self.env.total_nodes / max_level if self.env.levels else 0
        # Average Level Cost
        self.env.ALC = total_cost / max_level if max_level > 0 else 0

        self.env.nodes_in_thin_levels_mapping = self.find_nodes_in_thin_levels()
        self.update_levels_of_nodes_in_thin()
    

    # Finds the nodes in thin levels and returns a mapping to their actual node number, starting from 0
    def find_nodes_in_thin_levels(self):

        # Find thin levels
        self.env.thin_levels = [level for level, node_count in self.env.node_count_per_level.items() if node_count < self.env.ARL and self.env.level_costs[level] < self.env.ALC]
        self.env.thin_levels.sort()

        # Create a dictionary to map nodes to their indices
        nodes_in_thin_levels_mapping = {}
        index = 0
        for node, level in self.env.levels.items():
            if level in self.env.thin_levels:
                nodes_in_thin_levels_mapping[index] = node
                index += 1

        return nodes_in_thin_levels_mapping


    def init_levels_of_nodes_in_thin(self):
        levels_of_nodes_in_thin = {}
        indegrees_of_nodes_in_thin = {}
        for node, level in self.env.levels.items():
            if level in self.env.thin_levels:
                levels_of_nodes_in_thin[node] = self.env.levels[node]
                indegrees_of_nodes_in_thin[node] = self.env.indegree_dict[node]
        return levels_of_nodes_in_thin, indegrees_of_nodes_in_thin
    

    def update_levels_of_nodes_in_thin(self):
        for node in self.env.levels_of_nodes_in_thin.keys():
            self.env.levels_of_nodes_in_thin[node] = self.env.levels[node]
            self.env.indegrees_of_nodes_in_thin[node] = self.env.indegree_dict[node]
    

    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.total_nodes}\n"
            f"Total Levels: {max(self.env.levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.env.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.ARL:.2f}" )