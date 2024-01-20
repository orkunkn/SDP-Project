import networkx as nx
import matplotlib.pyplot as plt
from mtx_to_array import mtx_to_array

class Environment:
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.G = nx.DiGraph()

        # A dictionary mapping each node to its level.
        self.levels = {}

        # A dictionary mapping each node to its parent node.
        self.node_parents = {}

        # A dictionary to store initial levels of nodes, for reset purposes.
        self.init_levels = {}

        # A dictionary to store initial parent-child relationships, for reset purposes.
        self.init_parents = {}

        # A list to keep track of the current state of levels.
        self.state_levels = []

        # A dictionary mapping possible changable levels, their nodes and their indegrees.
        self.state_level_vectors = {}

        # A node counter for every level.
        self.nodes_per_level = {}

        # A dictionary for cost of every level
        self.level_costs = {}

        self.convert_matrix_to_graph()
        self.calculate_graph_metrics()
        
        
    """ Initial process, converts matrix to a graph """
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
            if row == col:
                self.G.add_edge(col, row)

        # Remove nodes without edges, except for node 0
        for x in range(1, self.matrix.shape[0]):
            if not self.G.out_degree(x) and not self.G.in_degree(x):
                self.G.remove_node(x)
                if x in self.node_parents:
                    del self.node_parents[x]
        
        self.init_levels = self.levels.copy()
        self.init_parents = self.node_parents.copy()
            
        return self.init_levels, self.init_parents

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):
        
        self.total_nodes = self.G.number_of_nodes()
        self.indegree_dict = {node: self.G.in_degree(node) for node in self.G.nodes()}
        # Calculate level costs
        self.level_costs = {}
        for node, level in self.levels.items():
            if level == 0:
                continue
            indegree = self.indegree_dict[node]
            cost = max(0, 2 * indegree - 1)

            if level in self.level_costs:
                self.level_costs[level] += cost
            else:
                self.level_costs[level] = cost

        # Count nodes in every level
        self.nodes_per_level = {}
        for node, level in self.levels.items():
            if level in self.nodes_per_level:
                self.nodes_per_level[level] += 1
            else:
                self.nodes_per_level[level] = 1
         
        total_parents = sum(self.indegree_dict.values())

        # 3 criterias to control
        # Average Indegree
        self.AIR = total_parents / self.total_nodes if self.total_nodes else 0
        # Average Row per Level
        self.ARL = self.total_nodes / (max(self.levels.values()) + 1) if self.levels else 0
        # Average Level Cost
        self.ALC = self.calculate_alc()

        # Check each node's calculated value against ALC and store their level and node with indegree
        for node in self.G.nodes():
            value = 2 * self.indegree_dict[node] - 1
            if value < self.ALC:
                node_level = self.levels[node]
                # Add node and its indegree to the respective level
                if node_level not in self.state_level_vectors:
                    self.state_level_vectors[node_level] = {}
                self.state_level_vectors[node_level][node] = self.indegree_dict[node]


    """ Graph drawing function """
    def draw_graph(self, info_text="",name=""):

        pos = {node: (node, -level) for node, level in self.levels.items()}
        plt.figure(figsize=(10, 8)) 
        nx.draw_networkx_nodes(self.G, pos, node_size=500)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), edge_color='black', arrows=True)
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")

        plt.text(0.005, 0.005, info_text, transform=plt.gca().transAxes, fontsize=13.5) 

        plt.title("Graph Representation with Levels")
        plt.axis("off")
        plt.savefig(f"{name}")
        plt.show()

    """ Moves a node to the highest level """
    def move_node_to_higher_level(self, node):
        if node not in self.G.nodes():
            print(f"Node {node} does not exist in the graph.")
            return

        original_level = self.levels[node]
        moved = False
        current_air = self.AIR
        current_alc = self.ALC
        current_arl = self.ARL
        print(self.generate_info_text())
    
        for new_level in range(original_level - 1, -1, -1):
            self.move_node(node, new_level)
            self.update_graph_after_movement(node, new_level)
            self.calculate_graph_metrics() # Recalculate metrics after moving the node

            new_air = self.AIR
            new_alc = self.calculate_alc()
            new_arl = self.ARL
            # if condition will be arranged
            if True:
                moved = True
                current_air, current_alc, current_arl = new_air, new_alc, new_arl
            else:
                print("not moved")
               
                self.G=self.create_graph_from_indegree(self.init_parents,self.init_levels)
                
                self.draw_graph(self.generate_info_text(),name="new_graph.png")
                print("------------")
                break

        if moved:
            print("Node moved successfully.")
            self.remove_empty_level(original_level)
            info = self.generate_info_text()
            self.draw_graph(info,name="new_graph.png")
            print(info)
              
        else:
            print(f"Node {node} did not improve metrics. It remains at level {original_level}.")
       
    """ Resets the graph """
    def create_graph_from_indegree(self, parent_dict, levels_dict):
        G = nx.DiGraph()
        # Add nodes to the graph along with their levels
        for node in parent_dict.keys():
            G.add_node(node, level=levels_dict[node])

        # Add edges based on the parent list
        for node, parents in parent_dict.items():
            for parent in parents:
                G.add_edge(parent, node)

        self.G = G
        self.levels = self.init_levels
        self.node_parents = self.init_parents
     
        return self.G


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


    def remove_empty_level(self, current_level):
        all_level = self.levels.values()
        self.remove_levels(current_level)
        
        if current_level not in all_level:
            for node,level in self.levels.items():
                if level > current_level:
                    self.levels[node]-=1
        else:
            return False


    def calculate_alc(self):
        """ Recalculate the Average Level Cost (ALC) after a node is moved. """
        ALC_numerator = sum(max(2 * indegree - 1, 0) for indegree in self.indegree_dict.values())
        return ALC_numerator / (max(self.levels.values()) + 1) if self.levels.values() else 0


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


""" basic_matrix.mtx

    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],

"""
matrix = mtx_to_array("basic_matrix.mtx")
env = Environment(matrix)
info_text = env.generate_info_text()
env.draw_graph(name="init_graph", info_text=info_text)
node_to_move = 5
env.move_node_to_higher_level(node_to_move)