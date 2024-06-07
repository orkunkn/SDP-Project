import scipy.io
import scipy.sparse
import networkx as nx
import os

upper_triangular = None

def mtx_to_array(file_name):
    global upper_triangular
    mtx_file = file_name  # path of the .mtx file
    sparse_matrix = scipy.io.mmread(mtx_file)

    # Extract the upper triangular part of the matrix
    lower_triangular = scipy.sparse.tril(sparse_matrix, -1)
    upper_triangular = scipy.sparse.triu(sparse_matrix, 0)
    # Convert to CSR format
    lower_triangular = lower_triangular.tocsr()

    return lower_triangular


def graph_to_mtx(graph, file_name):
    # Step 3: Convert the modified graph back to CSR matrix
    after_csr = nx.to_scipy_sparse_array(graph, format="csr", weight="weight")
    
    # Reintegrate the upper triangular part
    global upper_triangular
    if upper_triangular is not None:
        after_csr += upper_triangular

    folder = "mtx_files_after"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Optionally, save the new CSR matrix or perform further operations
    scipy.io.mmwrite(f"{folder}/{file_name}_after.mtx", after_csr)
    save_csr_components_as_bin(after_csr, prefix=file_name)


def save_csr_components_as_bin(csr_matrix, prefix='matrix'):
    folder = "bin_files"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the data, indices, and indptr components of the CSR matrix with .bin extension
    csr_matrix.data.tofile(f'bin_files/{prefix}_data.bin')
    csr_matrix.indices.tofile(f'bin_files/{prefix}_indices.bin')
    csr_matrix.indptr.tofile(f'bin_files/{prefix}_indptr.bin')