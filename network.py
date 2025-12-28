import random as r
import math as m

# Initialise the knowledge of weights and biases - input size is the length of the flattened list 
def init_network(input_size, output_size=26):
    # weights[0] will be a list of weights for 'A', weights[1] for 'B', etc.
    weights = [[r.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(output_size)]
    biases = [0.0 for _ in range(output_size)]

    return weights, biases

# Take a look at the image data and ask 26 experts (one for each letter) how much it looks like their specific letter
def forward(flat_list, weights, biases):
    # initalise all of the scores to 0 as we don't know anythign about the letters yet
    scores = [0.0] * len(weights)

    # Visit each letter expert
    for i in range(len(weights)):
        # start the letters score with the bias
        sum_val = biases[i]

        # Loop through the flat list and add the value * weight for that value to the sum
        for j in range(len(flat_list)):
            contribution = flat_list[j] * weights[i][j]
            sum_val += contribution

        # one we have checked all the pixels save the score for letter 'i' (var we are looping)
        scores[i] = sum_val

    # Return the list of 26 raw scrores, e.g [15.2, -3.1, 0.4 ...]
    return scores

# Since there is a large range of scores, if we attempt to find the exponent of the score the program could crash
# So we sqaush each score to ensure that it is between 0 and 1 and that the sum of the 26 results in 1.0
# This is done through some normalisation
def softmax(scores):
    # Find the biggets number in our 26 scores 
    max_score = max(scores)

    # Get list of each exponent score in its exponent version
    # we subtract the max_score to ge answers in the range 0-1 for the exponent of score
    exp_scores = [m.exp(score - max_score) for score in scores]

    # Get the sum of total weights
    sum_exp = sum(exp_scores)

    # Divide each experts individual value by the total sum to get a probability
    probs = [score/ sum_exp for score in exp_scores]

    return probs


