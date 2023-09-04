# External libraries
from numpy.linalg import det

from machine_learning_ii.utils import algebra as al


if __name__ == '__main__':

    matrix = al.create_matrix(4, 4)

    rank = al.rank_matrix(matrix)
    trace = al.trace_matrix(matrix)
    determinant = al.determinant_matrix(matrix)

    print(determinant == round(det(matrix)))

    print(f'Rank of matrix: {rank}')
    print(f'Trace of matrix: {trace}')
    print(f'Determinant of matrix: {determinant}')
