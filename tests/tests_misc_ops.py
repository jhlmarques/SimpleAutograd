import unittest
import numpy as np
from autograd.tensor import Tensor


class TestMiscOps(unittest.TestCase):
    def test_slice(self):
        t1 = Tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]], requires_grad=True)

        t2 = t1[0:2, ]
        t2.backward(np.array([[10, 11, 12], [13, 14, 15]]))

        assert t1.grad.tolist() == [[10, 11, 12], [13, 14, 15], [0, 0, 0]]
