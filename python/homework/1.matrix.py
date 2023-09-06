# External libraries
from numpy import trace
from numpy.linalg import det, matrix_rank

# Own libraries
from python.metadata.responses import Responses
from python.utils import algebra as al

if __name__ == '__main__':
    matrix = al.create_matrix(4, 4)
    print(f'Matrix:\n{matrix}')

    rank = al.rank_matrix(matrix)
    traces = al.trace_matrix(matrix)
    determinant = al.determinant_matrix(matrix)

    print(f'Rank of Matrix: {rank}')
    print(f'Trace of Matrix: {traces}')
    print(f'Determinant of Matrix: {determinant}')

    rank_match = rank == matrix_rank(matrix)
    trace_match = traces == trace(matrix)
    determinant_match = determinant == round(det(matrix))

    print(f'Rank Matches Numpy: {rank_match}')
    print(f'Trace Matches Numpy: {trace_match}')
    print(f'Determinant Matches Numpy: {determinant_match}')

    print(Responses.inverse_matrix)
    print(Responses.eigen)
