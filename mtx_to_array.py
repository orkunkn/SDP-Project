import scipy.io
import scipy.sparse

def mtx_to_array(file_name):
    # Load mtx file as a sparse matrix
    mtx_file = file_name  # path of the .mtx file
    sparse_matrix = scipy.io.mmread(mtx_file)

    # Extract the upper triangular part of the matrix
    lower_triangular = scipy.sparse.tril(sparse_matrix)

    # Convert it to a dense numpy array
    # Currently we can't convert a matrix bigger than 20000x20000 (approx) (needs too much RAM)
    dense_array = lower_triangular.toarray()
    return dense_array