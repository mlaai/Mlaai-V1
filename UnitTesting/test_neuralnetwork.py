import unittest
import Mlaai.V1 as mlaai

class Test_Mlaai_V1_NeuralNetwork(unittest.TestCase):
    def test_Neuralnetwork(self):
        config = '{"InputNodes": 1,"HiddenNodes": 2,"OutputNodes": 3,"LearningRate": 0.77}'
        network = mlaai.NeuralNetwork(config)

if __name__ == '__main__':
    unittest.main()
