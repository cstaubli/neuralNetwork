import numpy as np
import scipy.special


class NeuralNetwork(object):
    "my neural network"

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)
        )

        # learning rate
        self.learningrate = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    def train(self, inputs_list, targets_list):
        "train the neural network"
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        whodotmata = output_errors * final_outputs * (1.0 - final_outputs)
        whodotmatb = np.transpose(hidden_outputs)
        whodot = np.dot(whodotmata, whodotmatb)
        self.who += self.learningrate * whodot

        # update the weights for the links between the input and hidden layers
        wihdotmata = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        wihdotmatb = np.transpose(inputs)
        wihdot = np.dot(wihdotmata, wihdotmatb)
        self.wih += self.learningrate * wihdot

    def query(self, inputs_list):
        "query the neural network"
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
        "backquery the neural network"
        final_outputs = np.array(targets_list, ndmin=2).T
        final_inputs = self.inverse_activation_function(final_outputs)
        hidden_outputs = np.dot(self.who.T, final_inputs)

        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)

        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
