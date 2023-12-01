import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, matrix):
        if not self.is_lower_triangular(matrix):
            raise ValueError("Matrix must be lower triangular")
        self.matrix = matrix
        self.G = nx.DiGraph()
        self.levels = {}
        self.convert_matrix_to_graph()
        self.calculate_graph_metrics()

    def is_lower_triangular(self, matrix):
        """ Check if a matrix is lower triangular. """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        return np.all(matrix == np.tril(matrix))

    def convert_matrix_to_graph(self):
        """ Convert the matrix to a graph and calculate levels. """
        for x in range(self.matrix.shape[0]):
            self.G.add_node(x)
            node_has_edge = False
            self.levels[x] = 0
            for j in range(x):
                if self.matrix[x][j] == 1:
                    self.G.add_edge(j, x)
                    node_has_edge = True
                    self.levels[x] = max(self.levels[x], self.levels[j] + 1)
            if not node_has_edge and x != 0:  # Node 0 is kept as the root
                self.G.remove_node(x)

    def calculate_graph_metrics(self):
        """ Calculate and store graph metrics. """
        
        self.total_nodes = self.G.number_of_nodes()
        self.indegree_dict = {node: self.G.in_degree(node) for node in self.G.nodes()}
        self.total_levels = max(self.levels.values()) + 1
        
     
        total_parents = sum(self.indegree_dict.values())
        self.AIR = total_parents / self.total_nodes if self.total_nodes else 0
        
        ALC_numerator = sum(2 * indegree - 1 for indegree in self.indegree_dict.values())
        self.ALC = ALC_numerator / self.total_levels if self.total_levels else 0
        
        self.ARL = self.total_nodes / self.total_levels if self.total_levels else 0

    def draw_graph(self):
        """ Draw the graph. """
        pos = {node: (node, -level) for node, level in self.levels.items()}
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(self.G, pos, node_size=700)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), edge_color='black', arrows=True)
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")
        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.show()


""" We may consider to use random generated lower triangular matrix """
matrix = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 1]
])

env = Environment(matrix)
env.draw_graph()

print(f"Total Nodes: {env.total_nodes}")
print(f"Total Levels: {env.total_levels}")
print("Indegree (Parent Numbers) of Each Node:", env.indegree_dict)
print(f"Average Indegree per Row (AIR): {env.AIR:.2f}")
print(f"Average Level Cost (ALC): {env.ALC:.2f}")
print(f"Average Number of Rows per Level (ARL): {env.ARL:.2f}")
