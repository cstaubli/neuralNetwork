import numpy as np
from NeuralNetwork import NeuralNetwork
import scipy.misc

data_file_train = open("../../Downloads/mnist_train.csv", "r")
data_list_train = data_file_train.readlines()
data_file_train.close()

# self created image (png) with a number in it
img_array = scipy.misc.imread("../../Documents/two.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01

# number of input, hidden and output nodes
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10

for lr in range(2, 9, 1):
    # learning rate
    LEARNING_RATE = lr * 0.05

    # create instance of neural network
    nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # epochs is the number of times the training data set is used for training
    EPOCHS = 5

    for e in range(EPOCHS):
        for record in data_list_train:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(OUTPUT_NODES) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)
            pass
        pass

    outputs = nn.query(img_data)
    label = np.argmax(outputs)
    print(label)
    