import unittest
import numpy as np
import Mlaai.V1

class Test_Mlaai_V1_Helper(unittest.TestCase):
    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        y = 1/(1+ np.exp(-x))
        self.assertTrue(y[0] == (helper.sigmoid(x))[0])

    def test_sigmoidDerivative(self):
        x = np.array([1, 2, 3])
        y = 1/(1+ np.exp(-x))
        s = y*(1-y)
        self.assertTrue(s[0] == (helper.sigmoidDerivative(x))[0])

if __name__ == '__main__':
    unittest.main()
