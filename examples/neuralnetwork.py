from autograd import Tensor, exp
import numpy as np


def sigmoid(t: Tensor):
    return Tensor(1.) / (Tensor(1.) + exp(-t))


t_input = Tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.],
                  [0., 0.]])
t_expected = Tensor([[1.], [1.], [0.], [0.]])

t_weights1 = Tensor(np.random.randn(2, 2), requires_grad=True)
t_bias1 = Tensor([0.], requires_grad=True)
t_weights2 = Tensor(np.random.randn(2, 2), requires_grad=True)
t_bias2 = Tensor([0.], requires_grad=True)
t_weights3 = Tensor(np.random.randn(2, 1), requires_grad=True)
t_bias3 = Tensor([0.], requires_grad=True)

input_amount = 4
learning_rate = 0.3
batch_size = 1
epochs = 4000

print_epoch = True
for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0

    if print_epoch:
        print(f"Epoch {epoch}")

    for batch_start in range(0, input_amount, batch_size):
        t_weights1.zero_grad()
        t_bias1.zero_grad()
        t_weights2.zero_grad()
        t_bias2.zero_grad()
        t_weights3.zero_grad()
        t_bias3.zero_grad()

        batch_end = batch_start + batch_size
        t_batch = t_input[batch_start:batch_end]
        t_batch_expected = t_expected[batch_start:batch_end]

        t_output1 = sigmoid((t_batch @ t_weights1) + t_bias1)
        t_output2 = sigmoid((t_output1 @ t_weights2) + t_bias2)
        t_output3 = sigmoid((t_output2 @ t_weights3) + t_bias3)

        if print_epoch:
            print(f"\tBatch {batch_count}:\n\t\t"
                  f"Predicted = {t_output3.data.T}\n\t\tExpected  = {t_batch_expected.data.T}")
            batch_count += 1

        error = t_batch_expected - t_output3
        loss = (error ** 2).sum()

        loss.backward()
        epoch_loss += loss.data

        t_weights1 -= t_weights1.grad * learning_rate
        t_bias1 -= t_bias1.grad * learning_rate
        t_weights2 -= t_weights2.grad * learning_rate
        t_bias2 -= t_bias2.grad * learning_rate
        t_weights3 -= t_weights3.grad * learning_rate
        t_bias3 -= t_bias3.grad * learning_rate

    if print_epoch:
        print(f"Loss = {epoch_loss}")
        print_epoch = False

    if ((epoch + 1) % 100) == 0:
        print_epoch = True
