import sys
from imatrix import load_img
from imkernel import *
from draw import draw_matrix

if len(sys.argv) < 2:
    print("Usage: python main.py filename.png")
    sys.exit(1)

filename = sys.argv[1]

# load image, apply kernel adn draw the processed image
matrix = load_img(filename)
new = apply_kernel(matrix, edge)
draw_matrix(new, pixel_size=1, title="Processed image")
