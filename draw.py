import pygame as p

def draw_matrix(matrix, pixel_size=1, title="Image"):
    # Setup screen
    width = len(matrix[0])
    height = len(matrix)
    window = p.display.set_mode((width*pixel_size, height*pixel_size))
        
    running = True
    while running:
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False
                p.quit()

        # Draw the matrix
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                val = max(0, min(255, val))
                color = (val, val, val)  # grayscale
                p.draw.rect(window, color, (j*pixel_size, i*pixel_size, pixel_size, pixel_size))


        p.display.flip()