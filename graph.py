import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
class Graph:

    def __init__(self, environment):
        self.env = environment


    def convert_matrix_to_graph(self, matrix):
        rows, cols = matrix.nonzero()

        for x in range(matrix.shape[0]):
            self.env.G.add_node(x)
            self.env.node_levels[x] = 0

        for row, col in zip(rows, cols):
            if row > col:  # Ensuring lower triangular structure
                self.env.G.add_edge(col, row)
                self.env.node_levels[row] = max(self.env.node_levels[row], self.env.node_levels[col] + 1)

        self.env.node_move_count = defaultdict(int)
        self.env.node_count_per_level = defaultdict(int)
        self.env.level_costs = defaultdict(int)

        self.env.total_nodes = self.env.G.number_of_nodes()
        
        for node, level in self.env.node_levels.items():
            indegree = self.env.G.in_degree(node)
            cost = max(0, 2 * indegree - 1)
            
            # Update the level cost and total cost
            self.env.level_costs[level] += cost

            # Increment the node count for the level
            self.env.node_count_per_level[level] += 1


        """ Graph drawing function """
    def draw_graph(self, info_text="",name=""):

        pos = {node: (node, -level) for node, level in self.env.node_levels.items()}
        plt.figure(figsize=(10, 8)) 

        nx.draw_networkx_nodes(self.env.G, pos, node_size=70)
        nx.draw_networkx_edges(self.env.G, pos, edgelist=self.env.G.edges(), edge_color='black', arrows=True, arrowsize=4, width=0.2)
        nx.draw_networkx_labels(self.env.G, pos, font_size=5, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}")
        plt.show()
