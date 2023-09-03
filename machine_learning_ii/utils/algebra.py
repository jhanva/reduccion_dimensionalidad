# External libraries
import numpy as np


def create_matrix(rows: int, cols: int) -> np.array:
    """

    Args:
        rows:
        cols:

    Returns:

    """
    matrix = np.random.randint(100, size=(rows, cols))
    return matrix


def rank_matrix(arr: np.arr) -> int:
    """

    Args:
        arr:

    Returns:

    """
    rows, cols = arr.shape
    return min(rows, cols)


def trace_matrix(arr: np.array) -> int:
    """

    Args:
        arr:

    Returns:

    """
    trace = arr[0][0]
    for i in range(1, rank_matrix(arr)):
        trace = trace + arr[i][i]

    return trace


def det_matrix(arr: np.array):
    return None