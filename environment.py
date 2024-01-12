import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#
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
        """ Logical Equavilant of Lx=b """
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

    def find_level_by_node(node):
        """ Shows the node where it is located, at level """
        pass

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

    
    def draw_graph(self, info_text="",name=""):
        """ Draw the graph with additional information text. """
        info_text=self.generate_info_text()
        print(info_text)
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
        """ Move a node to a higher level if it reduces or maintains the ALC. 
            If the node moves to a level of its parent, break the edge and reassign grandparents as new parents. """
            
        if node not in self.G.nodes():
            print(f"Node {node} does not exist in the graph.")
            return

        current_alc = self.calculate_alc()
        print("current alc", current_alc)
        original_level = self.levels[node]
        moved = False
        move_counter=0
        while True:
            new_level = self.levels[node] - 1
            if new_level < 0:  # Node cannot go below level 0
                break
            
            # Temporarily move node to the new level
            self.levels[node] = new_level

            # Remove edges from parents that are now on the same level, and connect to grandparents
            parents_at_new_level = [parent for parent in self.G.predecessors(node) if self.levels[parent] == new_level]
            for parent in parents_at_new_level:
                self.G.remove_edge(parent, node)
                grandparents = list(self.G.predecessors(parent))
                for grandparent in grandparents:
                    self.G.add_edge(grandparent, node)

            # Check if ALC is maintained or improved
            new_alc = self.calculate_alc()
            if new_alc <= current_alc:
                current_alc = new_alc  # Update current ALC 
                moved = True
                move_counter+=1

            
            else:
                # Revert the changes
                self.levels[node] = new_level + 1
                for parent in parents_at_new_level:
                    self.G.add_edge(parent, node)
                    grandparents = list(self.G.predecessors(parent))
                    for grandparent in grandparents:
                        self.G.remove_edge(grandparent, node)
                break

        if moved:
            print("---------------- moved ----------------")
           
        else:
            print(f"Node {node} did not improve ALC. It remains at level {original_level}.")

    def calculate_alc(self):
        """ Recalculate the Average Level Cost (ALC) after a node is moved. """
        alc_numerator = sum(2 * self.indegree_dict[node] - 1 for node in self.G.nodes())
        alc_denominator = max(self.levels.values()) + 1  
        return alc_numerator / alc_denominator if alc_denominator else 0


    def nodes_vector(self,node):
        """ Writes a vector for each moved node in a graph """
        if not self.node.isnumeric():
            raise ValueError(f"Cant Move {self.node} The given node is not integer")

        if node > self.G.number_of_edges():
            raise ValueError("out of node number")


    def nodes_movement_watcher(self):
        """ This watcher will paire the random generated graph with the moved nodes and nodes's vectors """
        pass


    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.total_nodes}\n"
            f"Total Levels: {max(self.levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.calculate_alc():.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.total_nodes / (max(self.levels.values()) + 1):.2f}"
        )


""" will be our main source, sparse.tamu.edu """
matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],

])


env = Environment(matrix)
env.draw_graph(name="init_graph")
node_to_move=3
env.move_node_to_higher_level(node_to_move)
env.calculate_graph_metrics()
updated_info_text = env.generate_info_text()
env.draw_graph(info_text=updated_info_text,name="new_graph")



