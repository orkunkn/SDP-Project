import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Matrix in the papper (2021 // 3. page) 
matrix = np.array([
    [1, 0, 0, 0, 0, 0 ,0,0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0 ,0, 0],
    [1, 0, 0, 1, 0, 1 ,0, 0], 
    [0, 0, 1, 0, 0, 1 ,1, 0],
    [1, 0, 0, 1, 0, 0 ,1, 1]
])

# Create a directed graph
G = nx.DiGraph()


def convert_matrix_to_graph(matrix):
    levels = {}
    for x in range(matrix.shape[0]):
        # Add a node for each row (4x4) 4 nodes as default
        G.add_node(x)
        node_has_edge = False
        # Iterate over the columns in reverse for the lower triangular part
        levels[x] = 0
        for j in range(x):
            # If the matrix value is 1, add an edge from j to x
            if matrix[x][j] == 1:

                G.add_edge(j, x)
                #Nodes with edges are flagged
                node_has_edge = True
                levels[x] = max(levels[x], levels[j] + 1)
        #Unflagged edges are removed
        if not node_has_edge:
            G.remove_node(x)  
    return G, levels

G, levels=convert_matrix_to_graph(matrix)
# Define the position of each node using the levels
pos = {node: (node, -level) for node, level in levels.items()}

# Draw the graph
plt.figure(figsize=(10, 6))

# for nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# for edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrows=True)

# for labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

plt.title("Graph Representation with Levels")
plt.axis("off")  
plt.show()
