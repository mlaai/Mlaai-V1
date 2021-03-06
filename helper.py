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

    def initializeWithZeros(dim):
        """
        Initializes a vector to zero
        """
        return np.zeros((dim, 1)) 

    def propagate(w, b, X, Y):
        """
        w = weights
        b = bias
        X = data vector (x*y*z)
        Y = label vector
        """
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) * (-1/m)
        dw = np.dot(X, (A-Y).T) * (1/m)
        db = np.sum(A-Y) * (1/m)

        return {"dw": dw,
                "db": db}, cost

    def optimize(w, b, X, Y, numIterations, learningRate, printCost = False):
        """
        Runs gradient descent algorithm, optimizes w and b
        """
        costs = []
        for i in range(numIterations):
            grads, cost = propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learningRate * dw
            b = b - learningRate * db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                  "db": db}

        return params, grads, costs

    def predict(w, b, X):
        """
        Predict label 0 or 1 using learned logistic regression paramters: w and b
        """
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        A = sigmoid(np.dot(w.T,X) + b)

        for i in range(A.Shape[1]):
            if A[0, i] <= 0.5:
                Y_prediction[0, i] = 0
            else:
                Y_prediction[0, i] = 1

        return Y_prediction

    def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        """
        Builds the logistic regression model
        """

        # 1. Initialize weights and bias. 
        # Size is calculated form the Input Training dataset and weights are initialized to zeros for all the training dataset values
        # Bias is initialized to zero
        w = initializeWithZeros(X_train.shape[0])
        b = 0

        # 2. Calculate the gradient descent and costs
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = print_cost)
    
        w = parameters["w"]
        b = parameters["b"]

        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d