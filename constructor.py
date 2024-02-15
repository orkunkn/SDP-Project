import networkx as nx

class Constructor:
    
    def __init__(self, environment):
        self.env=environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):
        
        self.total_nodes = self.G.number_of_nodes()
        self.indegree_dict = {node: self.G.in_degree(node) for node in self.G.nodes()}
        
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
            f"Average Number of Rows per Level (ARL): {self.total_nodes / (max(self.levels.values()) + 1):.2f}" )