import numpy as np

class helper(object):
    """description of class"""
    def sigmoid(x):
        return 1/(1+ np.exp(-x))

    def sigmoidDerivative(x):
        return helper.sigmoid(x)*(1 - helper.sigmoid(x))