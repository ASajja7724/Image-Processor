import network as net
from imatrix import load_mnist_csv

import random as r

# training the model
def train():
    # step one -> load the training data
    print("Loading data ...")
    training_data = load_mnist_csv("mnist_train.csv")
    
    # initalise the network - 784 inputs for the 28x28 pixels, and 10 outputs
    weights, biases = net.initialise_network(784, 10)

    # set learning rate and number of passes
    learning_rate = 0.01
    repeats = 3

    for run in range(repeats):
        # shuffle data to prevent ovefitting
        r.shuffle(training_data)

        total_loss = 0
        correct_predictions = 0

        # loop trhough the trainig data
        for i, (label, pixels) in enumerate(training_data):
            # normalise the pixels to be in range 0-1 floats
            inputs = [pixel / 255.0 for pixel in pixels]

            # forward proppogate
            scores = net.forward_propogate(inputs, weights, biases)
            probs = net.softmax(scores)

            # Calculate the loss for this image for tracking purposes
            loss = net.calculate_error(probs, label)
            total_loss += loss

            # Check if prediction was correct
            prediction = probs.index(max(probs))
            if prediction == label: correct_predictions += 1

            # complete a backward pass and update weights
            dW, dB = net.backward_propogate(inputs, probs, label)
            weights, biases = net.update_parameteres(weights, biases, dW, dB, learning_rate)

            # output processes every 1000 images
            if i % 1000 == 0 and i > 0:
                avg_loss = total_loss / 1000
                accuracy = (correct_predictions / 1000) * 100
                print(f"Image {i}/{len(training_data)} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
                total_loss = 0
                correct_predictions = 0

    print("Training completed!")
    return weights, biases

if __name__ == "__main__":
    trained_weights, trained_biases = train()

    # save the weigths into a json file
    import json
    
    # We create a dictionary to hold both lists
    brain_data = {
        "weights": trained_weights, 
        "biases": trained_biases
    }
    
    with open("trained_params.json", "w") as f:
        json.dump(brain_data, f)
        
    print("Brain saved to trained_params.json!")
