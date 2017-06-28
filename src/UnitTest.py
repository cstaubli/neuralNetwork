import unittest
import numpy
from NeuralNetwork import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    "Unit test for NeuralNetwork"
    def setUp(self):
        self.inputs_list = [1.0, 0.5, -1.0]
        self.input_nodes = 3
        self.hidden_nodes = 3
        self.output_nodes = 3
        self.learning_rate = 0.3

    def test_query(self):
        "Test query method"
        mynn = NeuralNetwork(
            self.input_nodes,
            self.hidden_nodes,
            self.output_nodes,
            self.learning_rate)
        answer = mynn.query(self.inputs_list)
        check = isinstance(answer, numpy.ndarray)
        self.assertTrue(check)

if __name__ == '__main__':
    unittest.main()
