import helper as hp
import numpy as np

x = np.array([1, 2, 3])
print ("sigmoidDerivative(x) = " + str(hp.helper.sigmoidDerivative(x)))

config = '{"InputNodes": 1,"HiddenNodes": 2,"OutputNodes": 3,"LearningRate": 0.77}'
network = mlaai.NeuralNetwork(config)