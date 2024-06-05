class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):

        self.env.AIL = self.env.total_parents / self.env.level_count
        self.env.ARL = self.env.total_nodes / self.env.level_count
        self.env.ALC = self.env.total_cost / self.env.level_count
        
        # Condition to find thin levels or clean up based on thin levels array
        if len(self.env.thin_levels) >= self.env.first_thin_level_count * 0.3:
            self.find_thin_levels()
        else:
            # Removing empty levels
            mask = self.env.node_count_per_level[self.env.thin_levels] != 0
            self.env.thin_levels = self.env.thin_levels[mask]
        

    def find_thin_levels(self):

        node_counts = self.env.node_count_per_level[self.env.levels]
        level_costs = self.env.level_costs[self.env.levels]
        level_indegrees = self.env.level_indegrees[self.env.levels]

        # Conditions to identify thin levels based on metrics
        condition = (node_counts > 0) & (node_counts < self.env.ARL) & (level_costs < self.env.ALC) & (level_indegrees < self.env.AIL)

        # Extracting thin levels based on the condition
        thin_levels = self.env.levels[condition]
        thin_levels.sort()

        # Assign sorted thin levels to environment variable
        self.env.thin_levels = thin_levels

    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.G.number_of_nodes()}\n"
            f"Total Levels: {max(self.env.node_levels) + 1}\n"
            f"Indegree of Each Node: {dict(self.env.G.in_degree())}\n"
            f"Average Indegree per Level (AIL): {self.env.AIL:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.ARL:.2f}" )