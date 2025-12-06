# Kernel Matrices
identity = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]

sharpen = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

edge = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]

emboss = [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
]

blur = [
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
]

def apply_kernel(matrix, kernel_matrix=edge):
    changed_matrix = []
    # Removes some checks, increasing efficiency
    for i in range(len(matrix)-2): 
        row = []
        for j in range(len(matrix[0])-2):
            partial_matrix = [
                [matrix[i][j],   matrix[i][j+1],   matrix[i][j+2]],
                [matrix[i+1][j], matrix[i+1][j+1], matrix[i+1][j+2]],
                [matrix[i+2][j], matrix[i+2][j+1], matrix[i+2][j+2]],
            ]

            # Apply the kernel Matrix
            new_val = 0
            for a in range(len(kernel_matrix)):
                for b in range(len(kernel_matrix[0])):
                    new_val += (partial_matrix[a][b] * kernel_matrix[a][b])

            row.append(new_val)

        changed_matrix.append(row)

    return changed_matrix