import numpy as np
import scipy.special as sc
import matplotlib.pyplot as pp
from NeuralNetwork import NeuralNetwork

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_file_train = open("data/mnist_train_100.csv", "r")
data_list_train = data_file_train.readlines()
data_file_train.close()

data_file_test = open("data/mnist_test_10.csv", "r")
data_list_test = data_file_test.readlines()
data_file_test.close()

epochs = 5

for e in range(epochs):
    for record in data_list_train:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)
        pass
    pass

label = 9
targets = np.zeros(output_nodes) + 0.01
targets[label] = 0.99
print(targets)
image_data = nn.backquery(targets)
pp.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
pp.show()
