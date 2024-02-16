import networkx as nx
from graph import Graph
from constructor import Constructor


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

        self.AIR = 0
        self.ARL = 0
        self.ALC = 0

"""
matrix = mtx_to_array("bcsstk17.mtx")

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
"""