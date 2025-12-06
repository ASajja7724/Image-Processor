# The pooling section will be 2x2 
def max_pool(matrix, pool_size=2):
    pooled_vals = []

    # The modulus is used to prevent going out of range (i.e if the matrix cannot be split evenly by the pool siz)
    for i in range(0, len(matrix) - len(matrix) % pool_size, pool_size):
        for j in range(0, len(matrix[0]) - len(matrix[0]) % pool_size, pool_size):
            section = [
                matrix[i][j], matrix[i][j+1],
                matrix[i+1][j], matrix[i+1][j+1]
            ]

            # Use max pooling to get the max value
            max_val = max(section)

            pooled_vals.append(max_val)

    # reconstuct the pooled values into a matrix
    new_matrix = []
    count = 0
    for _ in range(len(matrix)//pool_size):
        row = []
        for _ in range(len(matrix[0])//pool_size):
            row.append(pooled_vals[count])
            count += 1

        new_matrix.append(row)

    return new_matrix

if __name__ == "__main__":
    test_matrix = [
        [1,2,3,4,5,6],
        [7,8,9,10,11,12],
        [13,14,15,16,17,18],
        [19,20,21,22,23,24],
        [25,26,27,28,29,30],
        [31,32,33,34,35,36]
    ]


    max_pool(test_matrix)
