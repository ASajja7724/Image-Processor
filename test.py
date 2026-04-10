import network as net
import imatrix as im
from draw import draw_matrix
import json
import random

def test_run():
    # load the weights and biases
    with open("trained_params.json", "r") as f:
        values = json.load(f)
        weights, biases = values["weights"], values["biases"]

    # load the test data
    test_data = im.load_mnist_csv("mnist_train.csv")

    #  pick a random number
    actual_label, pixels = random.choice(test_data)

    # Normalize by dividing by 255.0
    inputs = [p / 255.0 for p in pixels]
    scores = net.forward_propogate(inputs, weights, biases)
    probs = net.softmax(scores)

    # get the prediction,by getting the index of the highest probability
    prediction = probs.index(max(probs))

    # display results
    print(f"--- AI TEST ---")
    print(f"Actual Number: {actual_label}")
    print(f"AI Prediction: {prediction}")
    print("----------------")

    # display the image
    matrix_2d = im.reshape_to_2D(pixels, size=28)
    draw_matrix(matrix_2d, pixel_size=15, title=f"AI Guess: {prediction}")

if __name__ == "__main__":
    test_run()
