import numpy as np
class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):

        total_cost = sum(self.env.level_costs.values())
        level_count = max(self.env.node_levels.values()) + 1
        total_parents = sum(degree for _, degree in self.env.G.in_degree())

        self.env.AIL = total_parents / level_count
        self.env.ARL = self.env.total_nodes / level_count
        self.env.ALC = total_cost / level_count

        if len(self.env.thin_levels) >= self.env.first_thin_level_count * 0.1:
            self.find_thin_levels()


    def find_thin_levels(self):
        # Find thin levels
        thin_levels = [level for level, node_count in self.env.node_count_per_level.items()
                                if node_count < self.env.ARL and self.env.level_costs.get(level) < self.env.ALC and self.env.level_indegrees.get(level) < self.env.AIL]

        thin_levels.sort()
        self.env.thin_levels = np.array(thin_levels, dtype=int)


    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.G.number_of_nodes()}\n"
            f"Total Levels: {max(self.env.node_levels.values()) + 1}\n"
            f"Indegree of Each Node: {dict(self.env.G.in_degree())}\n"
            f"Average Indegree per Level (AIL): {self.env.AIL:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.ARL:.2f}" )