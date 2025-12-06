import sys
from imatrix import load_img
from imkernel import *
from draw import draw_matrix
from pool import max_pool

if len(sys.argv) < 2:
    print("Usage: python main.py filename.png")
    sys.exit(1)

filename = sys.argv[1]

# load image
matrix = load_img(filename)

# CNN LAYER 1

# apply edge kernel
layer1 = apply_kernel(matrix, edge)
# apply the pooling to the new image
layer1 = max_pool(layer1)

# apply edge kernel again
layer1 = apply_kernel(layer1, edge)


draw_matrix(layer1, pixel_size=1, title="Processed image")
