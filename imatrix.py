import pygame as p

p.init()
p.display.set_mode((1, 1))

# Load image
def load_img(image_path):
    image = p.image.load(image_path).convert()
    width, height = image.get_size()

    # Convert into a 2d greyscale matrix
    matrix = []

    # Loop through all of the pixels of the image
    for y in range(height):
        row = []
        for x in range(width):
            # Get colour -> get_at returns (R,G,B,A)
            r,g,b,_ = image.get_at((x,y))
            # Ink = 255, Paper = 0
            grey = 255 - ((r+g+b) // 3) 
            row.append(grey)

        matrix.append(row)

    return matrix

def load_mnist_csv(filename):
    # this will be tuple of the value and the pixels
    data = []
    with open (filename, "r") as f:
        # skip the first line as this is for lables
        next(f)
        for line in f:
            row = line.split(',')
            # the first digit it the number label
            label = int(row[0])
            # the second part is the pixels in a flattened view
            pixels = [int(pixel) for pixel in row[1:]]

            # add the values to the data
            data.append((label, pixels))

    return data

# reshaping a flat list to apply a kernel
def reshape_to_2D(flat_list, size=28):
    # MNIST has a list of 284 pixels (28 by 28 sqaure)
    matrix = []
    for i in range(0, len(flat_list), size):
        matrix.append(flat_list[i : i + size])

    return matrix