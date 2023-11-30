import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, matrix):
        if not self.is_lower_triangular(matrix):
            raise ValueError("Matrix must be lower triangular")
        self.matrix = matrix
        self.G = nx.DiGraph()

    def is_lower_triangular(self, matrix):
        """ Check if a matrix is lower triangular. """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        return np.all(matrix == np.tril(matrix))

    def convert_matrix_to_graph(self):
        levels = {}
        for x in range(self.matrix.shape[0]):
            self.G.add_node(x)
            node_has_edge = False
            levels[x] = 0
            for j in range(x):
                if self.matrix[x][j] == 1:
                    self.G.add_edge(j, x)
                    node_has_edge = True
                    levels[x] = max(levels[x], levels[j] + 1)

            if not node_has_edge:
                self.G.remove_node(x)
        
        return self.G, levels

    def draw_graph(self, G, levels):
        pos = {node: (node, -level) for node, level in levels.items()}
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.show()

# Example usage
matrix = np.array([
    [1, 0, 0, 0, 0, 0 ,0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0 ,0, 0],
    [1, 0, 0, 1, 0, 1 ,0, 0], 
    [0, 0, 1, 0, 0, 1 ,1, 0],
    [1, 0, 0, 1, 0, 0 ,1, 1]
])

env = Environment(matrix)
G, levels = env.convert_matrix_to_graph()
env.draw_graph(G, levels)
