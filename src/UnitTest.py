from NeuralNetwork import NeuralNetwork

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

MyNN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(MyNN.query([1.0, 0.5, -1.5]))
