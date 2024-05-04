
class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):

        total_cost = sum(self.env.level_costs.values())
        level_count = max(self.env.node_levels.values()) + 1
        total_parents = sum(dict(self.env.G.in_degree()).values())

        self.env.AIR = total_parents / self.env.total_nodes
        self.env.ARL = self.env.total_nodes / level_count
        self.env.ALC = total_cost / level_count

        self.find_thin_levels()


    def find_thin_levels(self):
        # Find thin levels
        self.env.thin_levels = [level for level, node_count in self.env.node_count_per_level.items()
                                if node_count < self.env.ARL and self.env.level_costs[level] < self.env.ALC]

        self.env.thin_levels.sort()


    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.G.number_of_nodes()}\n"
            f"Total Levels: {max(self.env.node_levels.values()) + 1}\n"
            f"Indegree of Each Node: {dict(self.env.G.in_degree())}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.ARL:.2f}" )