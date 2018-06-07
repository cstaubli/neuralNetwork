import numpy as np
from NeuralNetwork import NeuralNetwork
import scipy.misc
import time

data_file_train = open(name="../../Downloads/mnist_train.csv", mode="r")
data_list_train = data_file_train.readlines()
data_file_train.close()

# self created image (png) with a number in it
img_array = scipy.misc.imread(name="../../Documents/two.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01

# number of input, hidden and output nodes
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10

for lr in range(2, 9, 1):
    # learning rate
    LEARNING_RATE = lr * 0.05
    print("setting learning rate to {0}".format(LEARNING_RATE))

    # create instance of neural networkr
    nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # epochs is the number of times the training data set is used for training
    EPOCHS = 5

    for e in range(EPOCHS):
        start_time = time.time()
        for record in data_list_train:
            all_values = record.split(",")
            # adjust greyscale 0-255 to value from 0.01 to 1 and add 0.01 to prevent zero values
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(OUTPUT_NODES) + 0.01
            # set target number
            targets[int(all_values[0])] = 0.99
            # and train
            nn.train(inputs, targets)

        duration = time.time() - start_time
        print("run {0} is complete, duration was {1:0.2f} seconds".format(e, duration))

    # query with user picture
    outputs = nn.query(img_data)
    # get the output = result with max
    label = np.argmax(outputs)
    print(label)

