
class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):

        total_cost = sum(self.env.level_costs.values())
        level_count = max(self.env.node_levels.values(), default=0) + 1
        total_parents = sum(self.env.indegree_dict.values())

        self.env.AIR = total_parents / self.env.total_nodes if self.env.total_nodes else 0
        self.env.ARL = self.env.total_nodes / level_count if self.env.node_levels else 0
        self.env.ALC = total_cost / level_count if level_count > 0 else 0

        self.env.nodes_in_thin_levels_mapping = self.find_nodes_in_thin_levels()

        self.update_levels_of_nodes_in_thin()

    
    def find_nodes_in_thin_levels(self):
        # Find thin levels
        thin_levels = {level for level, node_count in self.env.node_count_per_level.items()
                    if node_count < self.env.ARL and self.env.level_costs[level] < self.env.ALC}
        
        self.env.thin_levels = thin_levels

        # Filter nodes that are in thin levels and map them to indices
        nodes_in_thin_levels = [node for node, level in self.env.node_levels.items() if level in thin_levels]
        nodes_in_thin_levels_mapping = {index: node for index, node in enumerate(nodes_in_thin_levels)}

        return nodes_in_thin_levels_mapping


    def init_levels_of_nodes_in_thin(self):
        levels_of_nodes_in_thin = {}
        indegrees_of_nodes_in_thin = {}
        for node, level in self.env.node_levels.items():
            if level in self.env.thin_levels:
                levels_of_nodes_in_thin[node] = self.env.node_levels[node]
                indegrees_of_nodes_in_thin[node] = self.env.indegree_dict[node]
        return levels_of_nodes_in_thin, indegrees_of_nodes_in_thin
    

    def update_levels_of_nodes_in_thin(self):
        for node in self.env.levels_of_nodes_in_thin.keys():
            self.env.levels_of_nodes_in_thin[node] = self.env.node_levels[node]
            self.env.indegrees_of_nodes_in_thin[node] = self.env.indegree_dict[node]
    

    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.G.number_of_nodes()}\n"
            f"Total Levels: {max(self.env.node_levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.env.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.ARL:.2f}" )