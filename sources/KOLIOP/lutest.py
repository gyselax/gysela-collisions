# [4., 0., 1., 0., 0., 0., 0., 0.],
# [0., 4., 0., 1., 0., 0., 0., 0.],
# [1., 0., 4., 0., 1., 0., 0., 0.],
# [0., 1., 0., 4., 0., 1., 0., 0.],
# [0., 0., 1., 0., 4., 0., 1., 0.],
# [0., 0., 0., 1., 0., 4., 0., 1.],
# [0., 0., 0., 0., 1., 0., 4., 0.],
# [0., 0., 0., 0., 0., 1., 0., 4.]]

# [4.   ,   0.   ,   1.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ],
# [0.   ,   4.   ,   0.   ,   1.   ,   0.   ,   0.   ,   0.   ,   0.   ],
# [0.   ,   0.   ,   3.75 ,   0.   ,   1.   ,   0.   ,   0.   ,   0.   ],
# [0.   ,   0.   ,   0.   ,   3.75 ,   0.   ,   1.   ,   0.   ,   0.   ],
# [0.   ,   0.   ,   0.   ,   0.   ,   3.73 ,   0.   ,   1.   ,   0.   ],
# [0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   3.73 ,   0.   ,   1.   ],
# [0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   3.73 ,   0.   ],
# [0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   3.73]]


# [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.
#  0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.
#  0.25 , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.
#  0.   , 0.25 , 0.   , 1.   , 0.   , 0.   , 0.   , 0.
#  0.   , 0.   , 0.26 , 0.   , 1.   , 0.   , 0.   , 0.
#  0.   , 0.   , 0.   , 0.26 , 0.   , 1.   , 0.   , 0.
#  0.   , 0.   , 0.   , 0.   , 0.26 , 0.   , 1.   , 0.
#  0.   , 0.   , 0.   , 0.   , 0.   , 0.26 , 0.   , 1.

# [4.   0.   1.   0.   0.   0.   0.   0.                        ]
# [0.   4.   0.   1.   0.   0.   0.   0.                        ]
# [0.25 0.   3.75 0.   1.   0.   0.   0.                        ]
# [0.   0.25 0.   3.75 0.   1.   0.   0.                        ]
# [0.   0.   0.26 0.   3.73 0.   1.   0.                        ]
# [0.   0.   0.   0.26 0.   3.73 0.   1.                        ]
# [0.   0.   0.   0.   0.26 0.   3.73 0.                        ]
# [0.   0.   0.   0.   0.   0.26 0.   3.7]]


import numpy as np
import scipy as sp
import scipy.linalg as la

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)

def ExtractLU(M):
    matrix_extent = len(M)
    L = np.zeros((matrix_extent, matrix_extent))
    U = np.zeros((matrix_extent, matrix_extent))
    for i in range(matrix_extent):
        L[i][i] = 1
        for j in range(matrix_extent):
            if j < i:
                L[i][j] = M[i][j]
            if j >= i:
                U[i][j] = M[i][j]
    return (L, U)

def ScalarLU(M):
    # TODO(Etienne M): investigate
    import copy
    M = copy.deepcopy(M)
    matrix_width = len(M)

    for pivot_idx in range(matrix_width):
        input_row_offset = pivot_idx + 1
        pivot_value_inv = 1.0 / M[pivot_idx][pivot_idx]

        for input_y in range(input_row_offset, matrix_width):
            row_scale_value = M[input_y][pivot_idx] * pivot_value_inv
            M[input_y][pivot_idx] = row_scale_value

            for input_x in range(input_row_offset, matrix_width):
                M[input_y][input_x] -= row_scale_value * M[pivot_idx][input_x]
    return M

# Tridiagonal matrix where scalars are dense small matrices.
def TridiagonalLU(M, block_extent):
    matrix_width = len(M)

    assert(matrix_width > 0)

    if matrix_width == 1:
        A_p = M[0][0]
        # (L,U) = sp.linalg.lu(A_p, permute_l=True)
        # # TODO(Etienne M): get the shape to produce the -I.
        # A_p = L + U - np.eye(block_extent)
        A_p = ScalarLU(A_p)
        M[0][0] = A_p
    else:
        for pivot in range(matrix_width - 1):
            # Assuming:
            # A_0 B_0 0
            # C_0 A_1 B_1
            # 0   C_1 A_2

            # For matrices as scalar:
            A_p = M[pivot][pivot]
            B_p = M[pivot][pivot + 1]
            C_p = M[pivot + 1][pivot]

            A_p1 = M[pivot + 1][pivot + 1]

            print(A_p)

            # NOTE: Pivoting sucks, we assume the matrices are diagonally
            # dominant.
            # (P, L, U) = sp.linalg.lu(A_p)

            (L, U) = ExtractLU(ScalarLU(A_p))

            A_p = L + U - np.eye(block_extent)

            print(A_p)

            B_p = np.linalg.inv(L) @ B_p
            C_p = C_p @ np.linalg.inv(U)

            A_p1 = A_p1 - (C_p @ B_p)

            M[pivot][pivot] = A_p
            M[pivot][pivot + 1] = B_p
            M[pivot + 1][pivot] = C_p
            M[pivot + 1][pivot + 1] = A_p1

        A_p = M[-1][-1]

        # (P, L, U) = sp.linalg.lu(A_p)
        # A_p = L + U - np.eye(block_extent)
        A_p = ScalarLU(A_p)

        M[-1][-1] = A_p

    return M

block_extent = 2
matrix_extent = 4

A = np.random.rand(block_extent, block_extent)
B = np.random.rand(block_extent, block_extent)
C = np.random.rand(block_extent, block_extent)

A = np.eye(block_extent) * 4 + A # Diagonally dominant so we prevent pivoting
B = np.eye(block_extent) * 1 + B # Diagonally dominant so we prevent pivoting
C = np.eye(block_extent) * 1 + C # Diagonally dominant so we prevent pivoting

Z = np.zeros((block_extent, block_extent))

M = [[A, B, Z, Z],
     [C, A, B, Z],
     [Z, C, A, B],
     [Z, Z, C, A]]

M_array = np.block(M)

print(M_array)

# ################################################################################

# (L, U) = sp.linalg.lu(M_array, permute_l=True)

################################################################################

# print(L + U - np.eye(matrix_extent * block_extent))
# print(L @ U)
# print(np.allclose(M_array - L @ U, np.zeros((block_extent * matrix_extent, block_extent * matrix_extent))))

################################################################################

print("TridiagonalLU")

Mlu_array = np.block(TridiagonalLU(M, block_extent))
(L, U) = ExtractLU(Mlu_array)

print(Mlu_array)
# print(L @ U)
print(np.allclose(M_array - L @ U, np.zeros((block_extent * matrix_extent, block_extent * matrix_extent))))

################################################################################

print("ScalarLU")

Mlu_array = ScalarLU(np.block(M_array))
(L, U) = ExtractLU(Mlu_array)

print(Mlu_array)
# print(L @ U)
print(np.allclose(M_array - L @ U, np.zeros((block_extent * matrix_extent, block_extent * matrix_extent))))

# diagonal_pattern = [C, B, A]

# M = []

# diagonal_on_sides = len(diagonal_pattern) // 2

# for i in range(matrix_extent):
#     a_row = []
#     for j in range(0, i - diagonal_on_sides):
#         a_row.append(0)

#     if i - diagonal_on_sides < 0:
#         pattern_first = diagonal_on_sides - i
#     else:
#         pattern_first = 0

#     pattern_first = max(i - diagonal_on_sides, 0)

#     pattern_first = max(i - diagonal_on_sides, i)
#     pattern_last  = min(i + diagonal_on_sides, matrix_extent) - diagonal_on_sides

#     for v in range(pattern_first, pattern_last):
#         a_row.append(diagonal_pattern[v])

#     print(pattern_first, pattern_last)

#     for j in range(i + diagonal_on_sides, matrix_extent):
#         a_row.append(0)

#     M.append(a_row)
