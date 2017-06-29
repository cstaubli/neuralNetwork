import numpy as np
import matplotlib.pyplot
# from NeuralNetwork import NeuralNetwork

data_file_train = open("data/mnist_train_100.csv", "r")
data_list_train = data_file_train.readlines()
data_file_train.close()

data_file_test = open("data/mnist_test_10.csv", "r")
data_list_test = data_file_test.readlines()
data_file_test.close()

first_line = data_list_train[0].split(',')
# image_array = np.asfarray(first_line[1:]).reshape((28, 28))
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation=None)
# matplotlib.pyplot.show()

# adjust greyscale 0-255 to value from 0.01 to 1
# add 0.01 to remove zero values 
scaled_input = (np.asfarray(first_line[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)
