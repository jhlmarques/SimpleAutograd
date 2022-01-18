# A differentiation engine, meant for usage in my neural network exercises
# Based on Joel Grus' "Livecoding an Autograd Library" (https://www.youtube.com/watch?v=RxmBukb-Om4)
# and the autograd library (https://github.com/HIPS/autograd)
#
# There are some attempts at explaining what's going on through comments; I'm sure that at least 50%
# of them are true

from typing import List, Union, NamedTuple, Callable, Optional
import numpy as np


# Each tensor can have a list of dependencies
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]
Tensorable = Union['Tensor', Arrayable]


# Converts, if needed, an Arrayable instance to a numpy array
def arrayable2array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


# Converts, if needed, a Tensorable instance to a Tensor
def tensorable2tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 dependencies: List[Dependency] = None) -> None:
        self.data = arrayable2array(data)
        self.requires_grad = requires_grad  # Basically means that this tensor will have it's gradient calculated
        self.dependencies = dependencies or []
        self.shape = self.data.shape
        self.grad: Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():  # No shape
                grad = np.array(1.0)
            else:
                raise RuntimeError("Grad must be specified for non-0 tensor")

        self.grad += grad

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def __add__(self, other) -> 'Tensor':
        return _add(self, tensorable2tensor(other))

    def __radd__(self, other) -> 'Tensor':
        return _add(tensorable2tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        self.data = self.data + tensorable2tensor(other).data
        return self

    def __sub__(self, other) -> 'Tensor':
        return _sub(self, tensorable2tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(tensorable2tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        self.data = self.data - tensorable2tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, tensorable2tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(tensorable2tensor(other), self)

    def __imul__(self, other) -> 'Tensor':
        self.data = self.data * tensorable2tensor(other).data
        return self

    def __truediv__(self, other) -> 'Tensor':
        return _div(self, tensorable2tensor(other))

    def __idiv__(self, other) -> 'Tensor':
        self.data = self.data / tensorable2tensor(other).data
        return self

    def __rdiv__(self, other) -> 'Tensor':
        return tensorable2tensor(other) / self.data

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __pow__(self, other) -> 'Tensor':
        return _pow(self, tensorable2tensor(other))

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, tensorable2tensor(other))

    def __getitem__(self, item) -> 'Tensor':
        return _slice(self, item)


# Returns a 0D tensor which is the sum of t's elements
def _tensor_sum(t: Tensor) -> Tensor:
    data = np.sum(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Grad is 0D because of the output;
            # For each element x, d Sum/ dX = 1
            return grad * np.ones_like(t.data)

        dependencies = [Dependency(t, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


# Handles numpy broadcasting conflicting with gradients
def handle_broadcasting(grad: np.ndarray, t: Tensor) -> np.ndarray:
    # Handles dimensions being added "to the beginning"
    # [1, 2] -> broadcast -> [[1, 2], [1, 2]] -> handle_broadcasting[2, 4]
    added_dimensions = grad.ndim - t.data.ndim
    for _ in range(added_dimensions):
        grad = np.sum(grad, axis=0)

    # Handles broadcasting in dimensions that are 1
    # Ex: (5 x 4 x 1) + (5 x 4 x 3) -> (5 x 4 x 3) + (5 x 4 x 3)
    for a, dim in enumerate(t.shape):
        # a = index = axis
        if dim == 1:
            grad = np.sum(grad, axis=a, keepdims=True)
    return grad


# Returns the sum of two tensors
def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    # Reasoning for grad_fn:
    # If f = t1 + t2, df/t1 = 1, df/t2 = 1
    # It also needs to consider numpy broadcasting; gradients relative to broadcast
    # elements should be summed up and then applied to the original tensor

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return handle_broadcasting(grad, t1)

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return handle_broadcasting(grad, t2)

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


# Multiplies two tensors
def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    # If f = a * b, df/da = b
    # Similar to sum, except the gradient is multiplied by b instead of 1

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            return handle_broadcasting(grad, t1)

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            return handle_broadcasting(grad, t2)

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


# Inverts the signals of a tensor's elements
def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        dependencies = [Dependency(t, lambda x: -x)]
    else:
        dependencies = []
    return Tensor(data, requires_grad, dependencies)


# Subtracts two tensors
def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return _add(t1, _neg(t2))


# Divides two tensors
def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * (1 / t2.data)
            return handle_broadcasting(grad, t1)

        dependencies.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * (-t1.data / np.square(t2.data))
            return handle_broadcasting(grad, t2)

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


# Raises the elements of a tensor to the power of the corresponding elements of another tensor
def _pow(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.power(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray):
            # If 
            grad = grad * (t2.data * np.power(t1.data, np.where(t2.data, t2.data - 1, 1)))
            return handle_broadcasting(grad, t1)

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray):
            grad = grad * np.log(np.where(t1.data, t1.data, 1)) * data
            return handle_broadcasting(grad, t2)

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


# Raises e to the power of every element of a tensor
def exp(t: Tensor) -> Tensor:
    data = np.exp(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad * data

        dependencies = [Dependency(t, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


# Applies log e to the elements of a tensor
def log(t: Tensor) -> Tensor:
    data = np.log(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad):
            return grad * (1 / t.data)

        dependencies = [Dependency(t, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


# Transposes a matrix
def transpose(t: Tensor) -> Tensor:
    data = t.data.T
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad.T
        dependencies = [Dependency(t, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


# Matrix multiplication
def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies = []

    # If T1 = M(m1 x n1) and T2 = M(m2 x n2), T1 @ T2 = M(m1 x n2) = grad's shape
    # This only works when n1 == m2; therefore:
    # T2.T = M(n2 x m2) = M(n2 x n1)
    # grad @ T2.T = M(m1 x n1)
    # and
    # T1.T = M(n1 x m1) = M(m2 x m1)
    # T1.T @ grad = M(m2 x n2)

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray):
            return grad @ t2.data.T

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray):
            return t1.data.T @ grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


# Slices a tensor
def _slice(t: Tensor, indexes: slice) -> Tensor:
    data = t.data[indexes]
    requires_grad = t.requires_grad

    # We want a matrix of the same shape of t.data, where the elements not
    # included in the slice are 0 (so when summed, they don't affect t's grad)

    if requires_grad:
        def grad_fn(grad: np.ndarray):
            data_grad = np.zeros_like(t.data)
            data_grad[indexes] = grad
            return data_grad
        dependencies = [Dependency(t, grad_fn)]

    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)
