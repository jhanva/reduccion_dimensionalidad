# External libraries
import numpy as np


def create_matrix(rows: int, cols: int) -> np.array:
    """Create a random matrix with the specified number of rows and columns.

    Args:
        rows: The number of rows in the matrix.
        cols: The number of columns in the matrix.

    Returns:
        A NumPy array representing the random matrix with values
         ranging from 0 to 99 (inclusive).

    """
    matrix = np.random.randint(100, size=(rows, cols))
    return matrix


def rank_matrix(matrix: np.array) -> int:
    """Calculate the rank of a 2D NumPy array by returning
     the minimum of its row and column counts.

    Args:
        matrix: A 2D NumPy array for which the rank needs to be determined.

    Returns:
        The rank of the input matrix, which is the minimum of its number
         of rows and columns.

    """
    rows, cols = matrix.shape
    return min(rows, cols)


def trace_matrix(matrix: np.array) -> int:
    """Calculate the trace (sum of diagonal elements) of a square 2D
     NumPy array.

    Args:
        matrix: A square 2D NumPy array for which the trace needs to be
         calculated.

    Returns:
        The trace of the input square matrix, which is the sum of its
         diagonal elements.

    """
    trace = matrix[0][0]
    for i in range(1, rank_matrix(matrix)):
        trace = trace + matrix[i][i]

    return trace


def calculate_cofactor(
    matrix: np.array, drop_row: int, drop_col: int
) -> np.array:
    """Calculate the cofactor of a given square 2D NumPy array by eliminating
     the specified row and column.

    Args:
        matrix: A square 2D NumPy array for which the cofactor needs to be
         calculated.
        drop_row: The index of the row to be eliminated.
        drop_col: The index of the column to be eliminated.

    Returns:
        np.array: The cofactor of the input square matrix after eliminating
         the specified row and column.

    """
    return [
        [matrix[i][j] for j in range(len(matrix[0])) if j != drop_col]
        for i in range(len(matrix))
        if i != drop_row
    ]


def determinant_matrix(matrix: np.array) -> int:
    """Calculate the determinant of a square 2D NumPy array.

    Args:
        matrixA square 2D NumPy array for which the determinant
         needs to be calculated.

    Returns:
        The determinant of the input square matrix.

    """
    # Get the number of rows and columns of the matrix
    n = len(matrix)

    # Base case: If the matrix is 1x1, the determinant is the only element
    if n == 1:
        return matrix[0][0]

    # Base case: If the matrix is 2x2, calculate the determinant directly
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant
    determinant = 0

    # Iterate through the first row to calculate the determinant
    for j in range(n):
        # Calculate the cofactor of the entry (0, j)
        cofactor = matrix[0][j] * determinant_matrix(
            calculate_cofactor(matrix, 0, j)
        )

        # Alternate the sign based on the column index
        if j % 2 == 1:
            cofactor = -cofactor

        determinant += cofactor

    return determinant
