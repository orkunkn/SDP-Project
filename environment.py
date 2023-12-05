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

    
    def draw_graph(self, info_text=""):
        """ Draw the graph with additional information text. """
        pos = {node: (node, -level) for node, level in self.levels.items()}
        plt.figure(figsize=(10, 8))  # Adjusted figure size for extra text space
        nx.draw_networkx_nodes(self.G, pos, node_size=500)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), edge_color='black', arrows=True)
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5)  # Display the info text

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig("graph.png")
        plt.show()


""" We may consider to use random generated lower triangular matrix """
matrix = np.array([
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1]
])

env = Environment(matrix)

info_text_ = (
    f"Total Nodes: {env.total_nodes}\n"
    f"Total Levels: {env.total_levels}\n"
    f"Indegree of Each Node: {env.indegree_dict}\n"
    f"Average Indegree per Row (AIR): {env.AIR:.2f}\n"
    f"Average Level Cost (ALC): {env.ALC:.2f}\n"
    f"Average Number of Rows per Level (ARL): {env.ARL:.2f}"
)

# Call the draw_graph method with the information text
env.draw_graph(info_text=info_text_)