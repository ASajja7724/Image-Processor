import random as r
import math as m

def initialise_network(input_size, output_size=10):
    # the weights var is a matrix which has dimensions outuput_size * input_size
    # this is initaly randomised to a very small weight, but will be updated as we train
    weights = [[r.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(output_size)]

    # biases: this is a list of length ouput_size, each of which can be changed as training occurs
    biases = [0.0 for _ in range(output_size)]

    return weights, biases

def forward_propogate(flat_list, weights, biases):
    #  This finds the score for each of the 10 nodes
    # the score is calculated using z = W.x + b, where W is the weight, x is the input, and b is the biase list (here a dot product is used)
    scores = []

    # iterate through each ouput neuron of the weights
    for i in range(len(weights)):
        # the inital score is the bias of this node
        score = biases[i]

        # calculate the dot product of the weights and input.
        # here a flattened version of the image makes this simpler to do
        for j in range(len(flat_list)):
            # loop through the input and multiply by the weight of this pixel for the current node to complete a dot product
            score += flat_list[j] * weights[i][j]

        scores.append(score)

    return scores

def softmax(scores):
    # here we use a sigmoid function (softmax activation funciton) to get probabilities
    max_score = max(scores)
    # here we use this variable to make sure that we dont reach errors when the score is large

    # find the e^score for all scores
    exp_scores = [m.exp[score-max_score] for score in scores]

    # find the sum of all the exponent - this is the demonator for all the probability calculations
    sum_exp = sum(exp_scores)

    # carry out the function for each score and store these as a list of probabilties
    probs = [(score/sum_exp) for score in exp_scores]

    return probs

