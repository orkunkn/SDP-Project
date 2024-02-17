import networkx as nx

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