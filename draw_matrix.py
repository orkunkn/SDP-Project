from mtx_conversions import mtx_to_array
import matplotlib.pyplot as plt
import os

def plot_matrix(csr):
    # Extract row and column indices from the CSR matrix
    row_indices, col_indices = csr.nonzero()

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(col_indices, row_indices, s=0.1)  # Adjust size of points if necessary
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix representation
    plt.title('Sparsity Pattern of the Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

"""
mtx_directory = "mtx_files"
for filename in os.listdir(mtx_directory):
    if filename.endswith(".mtx"):
        mtx_name = filename[:-4]  # Remove the '.mtx' extension
        print(mtx_name)
        matrix = mtx_to_array(f"{mtx_directory}/{filename}")

        plot_matrix(matrix)
"""
matrix = mtx_to_array(f"mtx_files/bcsstk37.mtx")
plot_matrix(matrix)