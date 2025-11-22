import pygame as p

p.init()

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
            grey = (r+g+b) // 3
            row.append(grey)

        matrix.append(row)

    return matrix