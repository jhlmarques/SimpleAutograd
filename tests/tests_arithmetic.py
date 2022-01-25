import unittest

import numpy as np

from autograd import Tensor


class TestArithmetic(unittest.TestCase):
    def test_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2
        t3.backward(np.array([3.4, 3.4, 3.4]))

        assert t1.grad.tolist() == [3.4, 3.4, 3.4]
        assert t2.grad.tolist() == [3.4, 3.4, 3.4]

    def test_add_broadcasting(self):
        t1 = Tensor([[1], [2], [3]], requires_grad=True)
        t2 = Tensor([[-1, -2, -3], [-4, -6, -8], [3, 5, 7]], requires_grad=True)

        t3 = t1 + t2
        t3.backward(np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]))

        assert t1.grad.tolist() == [[6], [6], [6]]
        assert t2.grad.tolist() == [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

    def test_mul(self):
        t1 = Tensor([[1, 2], [-1, -2]], requires_grad=True)
        t2 = Tensor([[4, 5], [6, 7]], requires_grad=True)

        t3 = t1 * t2
        t3.backward(np.array([[3, 1], [4, 2]]))

        assert t1.grad.tolist() == [[12, 5], [24, 14]]
        assert t2.grad.tolist() == [[3, 2], [-4, -4]]

    def test_mul_broadcasting(self):
        t1 = Tensor([[1], [2]], requires_grad=True)
        t2 = Tensor([[3, 4], [5, 6]], requires_grad=True)

        t3 = t1 * t2
        t3.backward(np.array([[9, 7], [8, 10]]))

        assert t1.grad.tolist() == [[55], [100]]
        assert t2.grad.tolist() == [[9, 7], [16, 20]]

    def test_neg(self):
        t1 = Tensor([1, 2, -4, 8], requires_grad=True)
        t2 = -t1

        t2.backward(np.array([5, 2, 3, 4]))

        assert t1.grad.tolist() == [-5, -2, -3, -4]

    def test_sub(self):
        t1 = Tensor([4, 5, 6], requires_grad=True)
        t2 = Tensor([9, 8, 7], requires_grad=True)

        t3 = t1 - t2
        t3.backward(np.array([1, 2, 3]))

        assert t1.grad.tolist() == [1, 2, 3]
        assert t2.grad.tolist() == [-1, -2, -3]

    def test_sub_broadcasting(self):
        t1 = Tensor([[1, 2]], requires_grad=True)
        t2 = Tensor([1], requires_grad=True)

        t3 = t1 - t2
        t3.backward(np.array([[3, 4]]))

        assert t1.grad.tolist() == [[3, 4]]
        assert t2.grad.tolist() == [-7]

    def test_div(self):
        t1 = Tensor([18, 16], requires_grad=True)
        t2 = Tensor([4, 3], requires_grad=True)

        t3 = t1 / t2
        t3.backward(np.array([5, 6]))

        assert t1.grad.tolist() == [5 / 4, 6 / 3]
        assert t2.grad.tolist() == [-90 / 16, -96 / 9]

    def test_div_broadcast(self):
        t1 = Tensor([[18, 16]], requires_grad=True)
        t2 = Tensor([4], requires_grad=True)

        t3 = t1 / t2
        t3.backward(np.array([[5, 6]]))

        assert t1.grad.tolist() == [[5 / 4, 6 / 4]]
        assert t2.grad.tolist() == [((-5 * 18) + (-6 * 16)) / 16]
