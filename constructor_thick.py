import json
import time

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
        
        # Process each node and level
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

        # Compute the 3 criteria
        self.env.AIR = total_parents / self.env.total_nodes if self.env.total_nodes else 0
        self.env.ARL = self.env.total_nodes / max_level if self.env.levels else 0
        self.env.ALC = total_cost / max_level if max_level > 0 else 0
    

    def calculate_total_grandparents(self, node):
        grandparents = set()
        for parent in self.env.G.predecessors(node):
            grandparents.update(self.env.G.predecessors(parent))

        return len(grandparents)
    

    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.total_nodes}\n"
            f"Total Levels: {max(self.env.levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.env.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.env.ALC:.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.total_nodes / (max(self.env.levels.values()) + 1):.2f}" )