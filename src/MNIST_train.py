import numpy as np
# import matplotlib.pyplot
from NeuralNetwork import NeuralNetwork

data_file_train = open("../../Downloads/mnist_train.csv", "r")
data_list_train = data_file_train.readlines()
data_file_train.close()

data_file_test = open("../../Downloads/mnist_test.csv", "r")
data_list_test = data_file_test.readlines()
data_file_test.close()

# number of input, hidden and output nodes
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10

for lr in range(1, 9, 1):
    # learning rate
    LEARNING_RATE = lr * 0.02

    # create instance of neural network
    nn = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

    # epochs is the number of times the training data set is used for training
    EPOCHS = 5

    for e in range(EPOCHS):
        for record in data_list_train:
            all_values = record.split(',')
            # adjust greyscale 0-255 to value from 0.01 to 1 and add 0.01 to prevent zero values
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(OUTPUT_NODES) + 0.01
            # set target number
            targets[int(all_values[0])] = 0.99
            # and train
            nn.train(inputs, targets)
        duration = time.time() - start_time
        print("run {0} is complete, duration was {1:0.2f} seconds".format(e, duration))

    SCORECARD = []

    for record in data_list_test:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = nn.query(inputs)
        label = np.argmax(outputs)
        if (label == correct_label):
            # print("correct, found {0}".format(label))
            SCORECARD.append(1)
        else:
            # print("wrong, found {0} excpected {1}".format(label, correct_label))
            SCORECARD.append(0)

    scorecard_array = np.asarray(SCORECARD)
    perf = scorecard_array.sum() / (scorecard_array.size * 1.0)
    print ("performance = {0:5f}, learning rate {1:1f}".format(perf, LEARNING_RATE))
