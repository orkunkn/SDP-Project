import scipy.io
import scipy.sparse
import numpy as np

def mtx_to_array(file_name):
    mtx_file = file_name  # path of the .mtx file
    sparse_matrix = scipy.io.mmread(mtx_file)

    # Extract the upper triangular part of the matrix
    lower_triangular = scipy.sparse.tril(sparse_matrix, -1)
    # Convert to CSR format
    lower_triangular = lower_triangular.tocsr()

    return lower_triangular

def graph_to_mtx(graph, file_name):
    # Determine the maximum node index to set the matrix size correctly
    max_node_index = max(graph.nodes()) + 1
    csr_matrix = scipy.sparse.lil_matrix((max_node_index, max_node_index), dtype=np.int32)

    # Iterate over all edges in the graph
    for u, v in graph.edges():
        csr_matrix[v, u] = 1  # Fill the matrix based on the directed edges

    # Convert lil_matrix to csr_matrix for efficient operations
    csr_matrix = csr_matrix.tocsr()
    scipy.io.mmwrite(f"{file_name}_after.mtx", csr_matrix)
