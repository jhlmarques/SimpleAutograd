import unittest
import numpy as np
import math
from autograd import Tensor, log, exp, transpose


class TestAlgebra(unittest.TestCase):

    def test_sum(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t1.sum()
        t1.backward([[2, 4, 6], [8, 10, 12]])

        assert t1.grad.tolist() == [[2, 4, 6], [8, 10, 12]]

    def test_sum_axis(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t1.sum(axis=1, keepdims=False)
        t1.backward([[2, 4, 6], [8, 10, 12]])

        assert t1.grad.tolist() == [[2, 4, 6], [8, 10, 12]]

    def test_sum_axis_keepdims(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t1.sum(axis=1, keepdims=True)
        t1.backward([[2, 4, 6], [8, 10, 12]])

        assert t1.grad.tolist() == [[2, 4, 6], [8, 10, 12]]

    def test_power(self):
        t1 = Tensor([2, 4, 5], requires_grad=True)
        t2 = Tensor([3, 2, 4], requires_grad=True)

        t3 = t1 ** t2
        t3.backward(np.array([6, 5, 7]))

        assert t1.grad.tolist() == [3 * (2 ** 2) * 6, 2 * (4 ** 1) * 5, 4 * (5 ** 3) * 7]
        assert t2.grad.tolist() == [6 * math.log(2) * (2 ** 3),
                                    5 * math.log(4) * (4 ** 2),
                                    7 * math.log(5) * (5 ** 4)]

    def test_power_zero1(self):
        t1 = Tensor([2, 4, 5], requires_grad=True)
        t2 = Tensor([0, 1, 2], requires_grad=True)

        t3 = t1 ** t2
        t3.backward(np.array([3, 4, 5]))

        assert t1.grad.tolist() == [0,
                                    1 * (4 ** 0) * 4,
                                    2 * (5 ** 1) * 5]
        assert t2.grad.tolist() == [3 * math.log(2) * (2 ** 0),
                                    4 * math.log(4) * (4 ** 1),
                                    5 * math.log(5) * (5 ** 2)]

    def test_power_zero2(self):
        t1 = Tensor([0, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 ** t2
        t3.backward(np.array([7, 8, 9]))

        assert t1.grad.tolist() == [0,
                                    8 * 5 * (2 ** 4),
                                    9 * 6 * (3 ** 5)]
        assert t2.grad.tolist() == [0,
                                    8 * math.log(2) * (2 ** 5),
                                    9 * math.log(3) * (3 ** 6)]

    def test_power_broadcasting(self):
        t1 = Tensor([2, 4, 5], requires_grad=True)
        t2 = Tensor([3], requires_grad=True)

        t3 = t1 ** t2
        t3.backward(np.array([6, 5, 7]))

        assert t1.grad.tolist() == [3 * (2 ** 2) * 6, 3 * (4 ** 2) * 5, 3 * (5 ** 2) * 7]
        assert t2.grad.tolist() == [6 * math.log(2) * (2 ** 3) +
                                    5 * math.log(4) * (4 ** 3) +
                                    7 * math.log(5) * (5 ** 3)]

    def test_exp(self):
        t1 = Tensor([6, 2, 0], requires_grad=True)

        t2 = exp(t1)
        t2.backward(np.array([1, 4, 7]))

        assert t1.grad.tolist() == [math.exp(6),
                                    4 * math.exp(2),
                                    7]

    def test_log(self):
        t1 = Tensor([6, 2, 7], requires_grad=True)

        t2 = log(t1)
        t2.backward(np.array([1, 4, 7]))

        assert t1.grad.tolist() == [1 / 6,
                                    4 / 2,
                                    1]

    def test_transpose1(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)

        t2 = transpose(t1)
        t2.backward(np.array([9, 8, 7]))

        assert t1.grad.tolist() == [9, 8, 7]

    def test_transpose2(self):
        t1 = Tensor([[1, 2, 3]], requires_grad=True)

        t2 = transpose(t1)
        t2.backward(np.array([[9], [8], [7]]))

        assert t1.grad.tolist() == [[9, 8, 7]]

    def test_matmul(self):
        t1 = Tensor([[1, 2],
                     [3, 4]], requires_grad=True)
        t2 = Tensor([[2, 3, 4],
                     [5, 6, 7]], requires_grad=True)

        t3 = t1 @ t2
        t3.backward(np.array([[1, 1, 1],
                              [2, 2, 2]]))

        assert t1.grad.tolist() == [[9, 18],
                                    [18, 36]]
        assert t2.grad.tolist() == [[7, 7, 7],
                                    [10, 10, 10]]
