import unittest
import numpy as np
import helper as hp

class Test_Helper(unittest.TestCase):
    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        y = 1/(1+ np.exp(-x))
        self.assertTrue(y[0] == (hp.helper.sigmoid(x))[0])

    def test_sigmoidDerivative(self):
        x = np.array([1, 2, 3])
        y = 1/(1+ np.exp(-x))
        s = y*(1-y)
        self.assertTrue(s[0] == (hp.helper.sigmoidDerivative(x))[0])

if __name__ == '__main__':
    unittest.main()
