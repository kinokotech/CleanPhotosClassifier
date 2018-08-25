# !/usr/bin/env python

from nose.tools import ok_, eq_
import numpy as np
import sys
sys.path.append('../src/')
import cnn

class TestCnn:

    def test_transform_onehot(self):
        one_hot = cnn.transform_onehot([1, 2, 3, 2])
        ys = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        eq_(np.allclose(one_hot, ys), True)
