import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mtx_to_array import mtx_to_array

class Environment:
    
    def __init__(self, matrix):

        self.matrix = matrix
        self.G = nx.DiGraph()
        self.levels = {}
        self.convert_matrix_to_graph()
        self.calculate_graph_metrics()


    def convert_matrix_to_graph(self):
        rows, cols = self.matrix.nonzero()

        for x in range(self.matrix.shape[0]):
            self.G.add_node(x)
            self.levels[x] = 0

        for row, col in zip(rows, cols):
            if row > col:  # Ensuring lower triangular structure
                self.G.add_edge(col, row)
                self.levels[row] = max(self.levels[row], self.levels[col] + 1)

        # Remove nodes without edges, except for node 0
        for x in range(1, self.matrix.shape[0]):
            if not self.G.out_degree(x) and not self.G.in_degree(x):
                self.G.remove_node(x)

    def is_level_empty(self,level):

        if level not in self.levels.values():
            return True
        else:
            False


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
        info_text=self.generate_info_text()
        print(info_text)

    
    def draw_graph(self, info_text="", name=""):
        """ Draw the graph with additional information text. """
        # print("levels draw", self.levels)
        pos = {node: (node, -level) for node, level in self.levels.items()}
        plt.figure(figsize=(10, 8)) 
        nx.draw_networkx_nodes(self.G, pos, node_size=500)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), edge_color='black', arrows=True)
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}.png")
        plt.show()

   
    def move_node_to_higher_level(self, node):
        if node not in self.G.nodes():
            print(f"Node {node} does not exist in the graph.")
            return

        original_level = self.levels[node]
        moved = False
        current_alc = self.calculate_alc()

        for new_level in range(original_level - 1, -1, -1):
            self.move_node(node, new_level)
            self.update_graph_after_movement(node, new_level)

            if self.calculate_alc() <= current_alc:
                current_alc = self.calculate_alc()
                moved = True
            else:
                self.revert_changes(node, original_level)
                break

        if moved:
            print("Node moved successfully.")

            if self.is_level_empty(original_level):
                self.remove_empty_level(original_level)
        else:
            print(f"Node {node} did not improve ALC. It remains at level {original_level}.")


    def move_node(self, node, new_level):
        self.levels[node] = new_level


    def update_graph_after_movement(self, node, new_level):
        # Remove edges from parents that are now on the same level
        for parent in list(self.G.predecessors(node)):
            if self.levels[parent] == new_level:
                self.G.remove_edge(parent, node)
                # Add edges from grandparents, if any
                for grandparent in self.G.predecessors(parent):
                    self.G.add_edge(grandparent, node)


    def remove_levels(self, level):
        keys_to_remove = [key for key, val in self.levels.items() if val == level]
        for key in keys_to_remove:
            del self.levels[key]


    def remove_empty_level(self, level):
        """ Shows where the node is located at which level """

        current_level=level
        all_level = self.levels.values()
        print(current_level)
        self.remove_levels(current_level)
        
        if current_level not in all_level:
            for node,level in self.levels.items():
                if level > current_level:
                    self.levels[node]-=1
        else:
            return False


    def calculate_alc(self):
        """ Recalculate the Average Level Cost (ALC) after a node is moved. """
        alc_numerator = sum(2 * self.indegree_dict[node] - 1 for node in self.G.nodes())
        alc_denominator = max(self.levels.values()) + 1  
        return alc_numerator / alc_denominator if alc_denominator else 0


    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.total_nodes}\n"
            f"Total Levels: {max(self.levels.values()) + 1}\n"
            # f"Indegree of Each Node: {self.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.calculate_alc():.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.total_nodes / (max(self.levels.values()) + 1):.2f}"
        )


matrix = mtx_to_array("basic_matrix.mtx")
env = Environment(matrix)
# env.draw_graph(name="init_graph")
node_to_move=3
env.move_node_to_higher_level(node_to_move)
env.calculate_graph_metrics()
# updated_info_text = env.generate_info_text()
# env.draw_graph(info_text=updated_info_text,name="new_graph")
