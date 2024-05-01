import scipy.io
import scipy.sparse

def mtx_to_array(file_name):
    mtx_file = file_name  # path of the .mtx file
    sparse_matrix = scipy.io.mmread(mtx_file)

    # Extract the upper triangular part of the matrix
    lower_triangular = scipy.sparse.tril(sparse_matrix, -1)
    # Convert to CSR format
    lower_triangular = lower_triangular.tocsr()

    return lower_triangular
