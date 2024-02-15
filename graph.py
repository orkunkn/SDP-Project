import matplotlib.pyplot as plt
import networkx as nx
class Graph:

    def __init__(self, environment):
        self.env=environment

    def convert_matrix_to_graph(self):
        rows, cols = self.matrix.nonzero()

        for x in range(self.matrix.shape[0]):
            self.G.add_node(x)
            self.levels[x] = 0

        for row, col in zip(rows, cols):
            if row > col:  # Ensuring lower triangular structure
                self.G.add_edge(col, row)
                self.levels[row] = max(self.levels[row], self.levels[col] + 1)

                # Add or update the parent list for each node
                if row in self.node_parents:
                    self.node_parents[row].append(col)
                else:
                    self.node_parents[row] = [col]
        
        self.init_levels = self.levels.copy()
        self.init_parents = self.node_parents.copy()
            
        return self.init_levels, self.init_parents


        """ Graph drawing function """
    def draw_graph(self, info_text="",name="",levels={}):

        pos = {node: (node, -level) for node, level in levels.items()}
        plt.figure(figsize=(10, 8)) 

        nx.draw_networkx_nodes(self.G, pos, node_size=80)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), edge_color='black', arrows=True, arrowsize=5, width=0.45)
        nx.draw_networkx_labels(self.G, pos, font_size=7, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}")
        plt.show()