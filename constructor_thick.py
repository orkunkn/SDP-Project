from collections import defaultdict

class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):

        self.env.total_nodes = self.env.G.number_of_nodes()
        self.env.indegree_dict = dict(self.env.G.in_degree())

        self.env.level_costs = defaultdict(int)
        self.env.node_count_per_level = defaultdict(int)

        total_cost = 0

        for node, level in self.env.levels.items():
            indegree = self.env.indegree_dict[node]
            cost = max(0, 2 * indegree - 1)
            
            # Update the level cost and total cost
            self.env.level_costs[level] += cost
            total_cost += cost
            
            # Increment the node count for the level
            self.env.node_count_per_level[level] += 1

        level_count = max(self.env.levels.values(), default=0) + 1
        total_parents = sum(self.env.indegree_dict.values())

        self.env.AIR = total_parents / self.env.total_nodes if self.env.total_nodes else 0
        self.env.ARL = self.env.total_nodes / level_count if self.env.levels else 0
        self.env.ALC = total_cost / level_count if level_count > 0 else 0
    

    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.total_nodes}\n"
            f"Total Levels: {max(self.env.levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.env.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.total_nodes / self.env.ARL:.2f}" )