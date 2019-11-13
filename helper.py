import numpy as np

class helper(object):
    """Helper class with common functions"""
    def sigmoid(x):
        """
        Compute the sigmoid of x
        """
        return 1/(1+ np.exp(-x))

    def sigmoidDerivative(x):
        """
        Compute the gradient of the sigmoid function with respect to its input x
        """
        s = helper.sigmoid(x)
        return s*(1 - s)

    def imageToVector(image):
        """
        Convert numpy array of shape (length, height, depth) to a vector of shape (length*height*depth, 1)
        """
        return image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))

    def normalizeRows(x):
        """
        Normalizes each row of matrix x to have a unit length
        """
        return x/np.linalg.norm(x, axis=1, keepdims=True)

    def softmax(x):
        """
        Convert a numpy matrix of shape (m,n) to a numpy matrix equal to the softmax of x, of shape (m,n)
        """ 
        xExp = np.exp(x)
        return xExp/np.sum(xExp, axis=1, keepdims=True)

    def L1(yHat, y):
        """
        Calculate Mean Absolute Error
        """
        return np.sum(np.absolute(yHat - y))

    def L2(yHat, y):
        """
        Calculate Mean Squared Error
        """
        return np.sum((yhat - y)**2)