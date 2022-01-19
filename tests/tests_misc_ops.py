import unittest
import numpy as np

import autograd
from autograd import Tensor


class TestMiscOps(unittest.TestCase):
    def test_slice(self):
        t1 = Tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]], requires_grad=True)

        t2 = t1[0:2, ]
        t2.backward(np.array([[10, 11, 12], [13, 14, 15]]))

        assert t1.grad.tolist() == [[10, 11, 12], [13, 14, 15], [0, 0, 0]]

    def test_maximum(self):
        t1 = Tensor([1, 3, -4, 5, 0.3, -2], requires_grad=True)

        t2 = autograd.maximum(t1, Tensor(0))
        t2.backward([5, 7, 9, 8, 3, 2])

        assert t1.grad.tolist() == [5, 7, 0, 8, 3, 0]

    def test_maximum2(self):
        t1 = Tensor([1, 3, -4, 5, 0.3, -2], requires_grad=True)

        t2 = autograd.maximum(t1, Tensor(3))
        t2.backward([5, 7, 9, 8, 3, 2])

        assert t1.grad.tolist() == [5, 7, 0, 8, 3, 0]