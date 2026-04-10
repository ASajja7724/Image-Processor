import network as net
import imatrix as im
import imkernel as imk
import pool as po
from draw import draw_matrix
import json
import random

def test_run():
    # load the weights and biases
    with open("trained_params.json", "r") as f:
        values = json.load(f)
        weights, biases = values["weights"], values["biases"]

    # load the test image
    image = im.load_img("my_digit.png")

    # Flatten the image
    pixels = []
    for row in image:
        for val in row:
            # normalise from 1 - 0
            pixels.append(val / 255.0)

    #  Predict (Make sure 'pixels' has exactly 784 items)
    if len(pixels) != 784:
        print(f"Error: Image size is wrong! Got {len(pixels)} pixels, need 784.")
        return

    scores = net.forward_propogate(pixels, weights, biases)
    probs = net.softmax(scores)

    # get the prediction,by getting the index of the highest probability
    prediction = probs.index(max(probs))

    # display results
    print(f"--- AI TEST ---\nAI Prediction: {prediction}\n----------------")
    
    confidence = max(probs) * 100
    print(f"AI Prediction: {prediction} ({confidence:.2f}% confidence)")


    draw_matrix(image, pixel_size=15, title=f"AI Guess: {prediction}")

if __name__ == "__main__":
    test_run()
