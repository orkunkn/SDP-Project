
class Constructor:
    
    def __init__(self, environment):
        self.env = environment

    """ Calculate and store graph metrics. """
    def calculate_graph_metrics(self):
        
        self.env.total_nodes = self.env.G.number_of_nodes()
        self.env.indegree_dict = {node: self.env.G.in_degree(node) for node in self.env.G.nodes()}
        
        """
        self.env.level_costs = {}
        for node, level in self.env.levels.items():
            if level == 0:
                continue
            indegree = self.env.indegree_dict[node]
            cost = max(0, 2 * indegree - 1)

            if level in self.env.level_costs:
                self.env.level_costs[level] += cost
            else:
                self.env.level_costs[level] = cost
        """

        # Count nodes in every level
        self.env.nodes_per_level = {}
        for node, level in self.env.levels.items():
            if level in self.env.nodes_per_level:
                self.env.nodes_per_level[level] += 1
            else:
                self.env.nodes_per_level[level] = 1
         
        total_parents = sum(self.env.indegree_dict.values())

        # 3 criterias to control
        # Average Indegree
        self.env.AIR = total_parents / self.env.total_nodes if self.env.total_nodes else 0
        # Average Row per Level
        self.env.ARL = self.env.total_nodes / (max(self.env.levels.values()) + 1) if self.env.levels else 0
        # Average Level Cost
        self.env.ALC = self.calculate_alc()

        """
        # Check each node's calculated value against ALC and store their level and node with indegree
        for node in self.env.G.nodes():
            value = 2 * self.env.indegree_dict[node] - 1
            if value < self.env.ALC:
                node_level = self.env.levels[node]
                # Add node and its indegree to the respective level
                if node_level not in self.env.state_level_vectors:
                    self.env.state_level_vectors[node_level] = {}
                self.env.state_level_vectors[node_level][node] = self.env.indegree_dict[node]
        """

    
    def update_graph_after_movement(self, node, new_level):
        # Remove edges from parents that are now on the same level
        for parent in list(self.env.G.predecessors(node)):
            if self.env.levels[parent] == new_level:
                self.env.G.remove_edge(parent, node)
                # Add edges from grandparents, if any
                for grandparent in self.env.G.predecessors(parent):
                    self.env.G.add_edge(grandparent, node)


    def calculate_total_grandparents(self, node):
        grandparents = set()
        for parent in self.env.G.predecessors(node):
            grandparents.update(self.env.G.predecessors(parent))

        return len(grandparents)


    def get_by_thin_levels(self):
        for levels in self.env.state_level_vectors:
            sub_dict = self.env.state_level_vectors[levels]

            # Calculate the frequency of each value
            value_frequencies = {}
            for value in sub_dict.values():
                if value in value_frequencies:
                    value_frequencies[value] += 1
                else:
                    value_frequencies[value] = 1

            # Sorting each inner dictionary by the frequency of its values (from lowest to highest)
            sorted_sub_dict = dict(sorted(sub_dict.items(), key=lambda item: value_frequencies[item[1]]))
            self.env.state_level_vectors[levels] = sorted_sub_dict

        return self.env.state_level_vectors

        
    def calculate_alc(self):
        """ Recalculate the Average Level Cost (ALC) after a node is moved. """
        ALC_numerator = sum(max(2 * indegree - 1, 0) for indegree in self.env.indegree_dict.values())
        return ALC_numerator / (max(self.env.levels.values()) + 1) if self.env.levels.values() else 0


    def generate_info_text(self):
        """ Generate information text about the graph metrics. """
        return (
            f"Total Nodes: {self.env.total_nodes}\n"
            f"Total Levels: {max(self.env.levels.values()) + 1}\n"
            f"Indegree of Each Node: {self.env.indegree_dict}\n"
            f"Average Indegree per Row (AIR): {self.env.AIR:.2f}\n"
            f"Average Level Cost (ALC): {self.calculate_alc():.2f}\n"
            f"Average Number of Rows per Level (ARL): {self.env.total_nodes / (max(self.env.levels.values()) + 1):.2f}" )