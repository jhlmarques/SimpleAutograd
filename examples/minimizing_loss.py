import autograd
from autograd import Tensor

y_hat1 = Tensor([3, 6.1, 9.8], requires_grad=True)
y1 = Tensor([2, 1, 4])

# MSE
print("Mean Squared Error")
for i in range(20):
    y_hat1.zero_grad()
    print(f"y_hat: {y_hat1}")
    loss = ((y_hat1 - y1) ** 2).sum() / 3
    print(f"Loss: {loss.data}")
    loss.backward()
    y_hat1.data -= y_hat1.grad * 0.1

y_hat2 = Tensor([0.45, 0.15, 0.4], requires_grad=True)
y2 = Tensor([1, 0, 0])

# Cross-Entropy Loss
print("\nCross Entropy Loss:")
for i in range(20):
    y_hat2.zero_grad()
    print(f"y_hat: {y_hat2}")
    loss = (-(((y2 * autograd.log(y_hat2)) + ((1 - y2) * autograd.log((1 - y_hat2)))).sum())) / 3
    print(f"Loss: {loss.data}")
    loss.backward()
    y_hat2.data -= y_hat2.grad * 0.1

