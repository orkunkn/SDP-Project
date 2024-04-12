import matplotlib.pyplot as plt
import networkx as nx

class Graph:

    def __init__(self, environment):
        self.env = environment


    def convert_matrix_to_graph(self, matrix):
        rows, cols = matrix.nonzero()

        for x in range(matrix.shape[0]):
            self.env.G.add_node(x)
            self.env.levels[x] = 0

        for row, col in zip(rows, cols):
            if row > col:  # Ensuring lower triangular structure
                self.env.G.add_edge(col, row)
                self.env.levels[row] = max(self.env.levels[row], self.env.levels[col] + 1)


        """ Graph drawing function """
    def draw_graph(self, info_text="",name=""):

        pos = {node: (node, -level) for node, level in self.env.levels.items()}
        plt.figure(figsize=(10, 8)) 

        nx.draw_networkx_nodes(self.env.G, pos, node_size=80)
        nx.draw_networkx_edges(self.env.G, pos, edgelist=self.env.G.edges(), edge_color='black', arrows=True, arrowsize=5, width=0.45)
        nx.draw_networkx_labels(self.env.G, pos, font_size=7, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}")
        plt.show()
