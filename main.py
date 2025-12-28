import sys
from imatrix import load_img
from imkernel import *
from draw import draw_matrix
from pool import max_pool
from network import init_network, forward, softmax

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

# CNN LAYER 2

# apply sharpen kernel
layer2 = apply_kernel(layer1, sharpen)
# apply pooling again
layer2 = max_pool(layer2)

# Flatten the resultant matrix
flat = []
for row in layer2:
    for val in row:
        # normalise from 1 - 0
        flat.append(val / 255.0)

# Show just length or first few items
print("Length of flat vector:", len(flat))
print("First 50 elements:", flat[:50])

# Initialise the network
W, B = init_network(len(flat), 26)

# Get the raw scores form the Experts
raw_scores = forward(flat, W, B)

# Use softwax to get the confidence
probs = softmax(raw_scores)

# Find the best index
best_i = probs.index(max(probs))
confidence = probs[best_i]

# Output the answer
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print(f"I am {confidence*100:.1f}% sure this is the letter {alphabet[best_i]}")

draw_matrix(layer2, pixel_size=1, title="Processed image")
