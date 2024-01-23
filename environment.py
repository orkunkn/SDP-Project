import networkx as nx
import matplotlib.pyplot as plt
from mtx_to_array import mtx_to_array
import sys


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

    """ Moves a node to the highest level """
    def move_node_to_higher_level(self, node):
        if node not in self.G.nodes():
            print(f"Node {node} does not exist in the graph.")
            return

        original_level = self.levels[node]
        moved = False
        current_air = self.AIR

        for new_level in range(original_level - 1, -1, -1):
            # Count grandparents for the new level
            grandparents_count = self.calculate_total_grandparents(node)
          
            if grandparents_count > current_air:
                self.move_node(node, new_level)
                self.update_graph_after_movement(node, new_level)
                self.calculate_graph_metrics()  
                moved = True
                print(f"Node {node} moved to level {new_level}.")
                break
            else:
                
                self.G=self.create_graph_from_indegree(self.init_parents,self.init_levels)
                print(f"Node {node} not moved, grandparents count ({grandparents_count}) is not greater than AIR ({current_air}).")
                break
               

        if moved:
            x=self.remove_empty_level(original_level)
            print("Node moved successfully.", node)
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


    def calculate_total_grandparents(self, node):
        grandparents = set()
        for parent in self.G.predecessors(node):
            grandparents.update(self.G.predecessors(parent))
        return len(grandparents)



    def remove_levels(self, level):
        keys_to_remove = [key for key, val in self.levels.items() if val == level]
    
        for key in keys_to_remove:
            del self.levels[key]


    def remove_empty_level(self, current_level):
        all_level = self.levels.values()
        print(current_level not in all_level)
        if current_level not in all_level:
            self.remove_levels(current_level)
            for node,level in self.levels.items():
                if level > current_level:
                    self.levels[node]-=1
        else:
            return False
     
    def move_nodes_from_small_levels(self):
        level_counts = {level: 0 for level in set(self.levels.values())}
        for node in self.G.nodes():
            level_counts[self.levels[node]] += 1

        for level, count in level_counts.items():
            if count in [1, 2, 3]:
                # Move all nodes from this level to the next higher level
                for node in [n for n, lvl in self.levels.items() if lvl == level]:
                    new_level = level - 1
                    self.move_node(node, new_level)
                    self.update_graph_after_movement(node, new_level)
                self.calculate_graph_metrics()
                self.remove_empty_level(level)
                print(f"Nodes from level {level} moved to level {new_level}.")
 

    def get_by_thin_levels(self):
        for levels in self.state_level_vectors:
            sub_dict = self.state_level_vectors[levels]

            # Calculate the frequency of each value
            value_frequencies = {}
            for value in sub_dict.values():
                if value in value_frequencies:
                    value_frequencies[value] += 1
                else:
                    value_frequencies[value] = 1

            # Sorting each inner dictionary by the frequency of its values (from lowest to highest)
            sorted_sub_dict = dict(sorted(sub_dict.items(), key=lambda item: value_frequencies[item[1]]))
            self.state_level_vectors[levels] = sorted_sub_dict

        return self.state_level_vectors

        
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



matrix = mtx_to_array("bcsstk01.mtx")

env = Environment(matrix)
info_text = env.generate_info_text()

log_file = open("graph_processing_log.txt", "w")
sys.stdout = log_file



ordered_level_states=env.get_by_thin_levels()

env.draw_graph(name="first", info_text=info_text, levels=env.levels)
for states in reversed(list(ordered_level_states.keys())):
    node_states=ordered_level_states[states]
    sorted_tuples = sorted(node_states.items(), key=lambda item: item[1])
    sorted_node_states = dict(sorted_tuples)
 
    for node in sorted_node_states.keys():
      
        info_text = env.generate_info_text()
        env.move_node_to_higher_level(node)
        

node_parents=env.node_parents
levels=env.levels
        #
for node, parents in list(node_parents.items()):
        node_level = levels[node]
        node_parents[node] = [parent for parent in parents if levels[parent] <= node_level]
        if node_level == 0:
            
            node_parents[node] = []

env.create_graph_from_indegree(node_parents, levels)
env.move_nodes_from_small_levels()
env.draw_graph(name="tester", info_text=env.generate_info_text(), levels=env.levels)

sys.stdout = sys.__stdout__
log_file.close()
