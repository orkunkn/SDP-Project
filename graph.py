import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
class Graph:

    def __init__(self, environment):
        self.env = environment


    def convert_matrix_to_graph(self, matrix):
        rows, cols = matrix.nonzero()
        data = matrix.data
        self.env.G.add_nodes_from(range(matrix.shape[0]))
        self.env.node_levels = np.zeros(matrix.shape[0], dtype=int)
        for row, col, weight in zip(rows, cols, data):
            self.env.G.add_edge(col, row, weight=weight)
            self.env.node_levels[row] = max(self.env.node_levels[row], self.env.node_levels[col] + 1)

        max_level = np.max(self.env.node_levels) + 1  # Find max level for array sizes

        # Initialize NumPy arrays
        self.env.node_move_count = np.zeros(len(self.env.node_levels), dtype=int)
        self.env.node_count_per_level = np.zeros(max_level, dtype=int)
        self.env.level_costs = np.zeros(max_level, dtype=int)
        self.env.level_indegrees = np.zeros(max_level, dtype=int)
        self.env.levels = np.unique(self.env.node_levels)

        # Compute in-degrees for all nodes
        in_degrees = dict(self.env.G.in_degree())

        # Precompute the sum of in-degrees of parent nodes for each node
        sum_costs_of_parents = [sum(2 * in_degrees[pred] - 1 for pred in self.env.G.predecessors(node)) for node in self.env.G.nodes()]

        # Convert the results into a numpy array
        self.env.node_parents_cost_sum = np.array(sum_costs_of_parents)
        
        self.env.total_nodes = self.env.G.number_of_nodes()
        self.env.total_parents = sum(degree for _, degree in self.env.G.in_degree())
        self.env.level_count = len(self.env.levels)
            
        for node in range(len(self.env.node_levels)):
            level = self.env.node_levels[node]
            indegree = self.env.G.in_degree(node)
            cost = max(0, 2 * indegree - 1)

            # Update the level cost and total cost
            self.env.level_costs[level] += cost
            self.env.total_cost += cost

            # Increment the node count for the level
            self.env.node_count_per_level[level] += 1

            # Update the level indegrees
            self.env.level_indegrees[level] += indegree

        """ Graph drawing function """
    def draw_graph(self, info_text="",name=""):

        pos = {node: (node, -self.env.node_levels[node]) for node in range(len(self.env.node_levels))}

        plt.figure(figsize=(10, 8)) 

        nx.draw_networkx_nodes(self.env.G, pos, node_size=70)
        nx.draw_networkx_edges(self.env.G, pos, edgelist=self.env.G.edges(), edge_color='black', arrows=True, arrowsize=4, width=0.2)
        nx.draw_networkx_labels(self.env.G, pos, font_size=5, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}")
        plt.show()
