import numpy as np
import json

class Config(object):
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

    def objectDecoder(obj):
        if '__type__' in obj and obj['__type__'] == 'Config':
            return Config(obj['InputNodes'], obj['HiddenNodes'], obj['OutputNodes'])
        return obj

class NeuralNetwork(object):
    def __init__(self, config):
        self.__dict__ = json.loads(config)
        cf = Config(**self.__dict__)

        self.weightsInputToHidden = np.random.normal(0.0, 
                                                    cf['InputNodes']**-0.5,
                                                    (cf['InputNodes'], 
                                                        cf['HiddenNodes']))
        
        self.weightsHiddenToOutput = np.random.normal(0.0, 
                                                        cf['HiddenNodes']**-0.5, 
                                                        (cf['HiddenNodes'], 
                                                           cf['OutputNodes']))

        self.activationFunction = lambda x : 1/(1+np.exp(-x))
        
    def Train(self, features, targets):
        records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weightsInputToHidden.shape) #returns new array with filled-in zeroes
        delta_weights_h_o = np.zeros(self.weightsHiddenToOutput.shape) #returns new array with filled-in zeroes
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.ForwardPass(X)
            delta_weights_i_h, delta_weights_h_o = self.Backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.UpdateWeights(delta_weights_i_h, delta_weights_h_o, records)

    def ForwardPass(self, features):
        hidden_inputs = np.dot(features, self.weightsInputToHidden)

        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weightsHiddenToOutput)

        final_outputs = final_inputs

        return final_outputs, hidden_outputs


    def Backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = y - final_outputs 

        output_error_term = error * 1.0 

        hidden_error = np.dot(self.weightsHiddenToOutput, output_error_term)

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        delta_weights_i_h += hidden_error_term * X[:, None]

        delta_weights_h_o += output_error_term * hidden_outputs[:,None]

        return delta_weights_i_h, delta_weights_h_o

    def UpdateWeights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weightsHiddenToOutput += self.__dict__.learningRate * delta_weights_h_o / n_records  
        self.weightsInputToHidden += self.__dict__.learningRate * delta_weights_i_h / n_records

    def Run(self, features):
        hidden_inputs = np.dot(features, self.weightsInputToHidden)
        hidden_outputs = self.activationFunction(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weightsHiddenToOutput)
        final_outputs = final_inputs
        return final_outputs

