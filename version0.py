"""
version0.py: linear regression with hand-derived gradients in numpy.

What this file teaches
- The data is a noisy quadratic, y = 5 - 10 x^2 - 3 x + noise. A linear model
  can still fit it if we hand it the feature x^2: inputs become [x, x^2].
- The model is a dict of parameters (weights, bias); forward() is x W + b.
- The loss is L = 1/2 mean((y_pred - y)^2), and we derive its gradients by
  hand: dL/dW = X^T (y_pred - y) / N and dL/db = mean(y_pred - y).
- Full-batch gradient descent: every update uses the whole training set.

New compared to the previous file: this is the first file. The same model is
rewritten step by step until it becomes a Flax/Optax CNN in version8.py.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import generate_data, get_parser, set_seed, train_test_split


def get_params(fan_in, fan_out):
    return dict(
        weights=np.random.randn(fan_in, fan_out) / np.sqrt(fan_in + fan_out),
        bias=np.zeros((fan_out,))
    )


def loss_fn(y_pred, y_true):
    return 0.5 * np.mean((y_pred - y_true) ** 2)


def forward(params, x):
    return np.dot(x, params['weights']) + params['bias']


def update(params, x, y, lr=0.01):
    y_pred = forward(params, x)

    # gradients of 1/2 mean((x W + b - y)^2), derived by hand
    grad_weights = np.dot(x.T, (y_pred - y)) / x.shape[0]
    grad_bias = np.mean(y_pred - y, axis=0)

    params['weights'] -= lr * grad_weights
    params['bias'] -= lr * grad_bias

    return params


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']

    set_seed(seed)

    x, y = generate_data()
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    # feature engineering: [x] -> [x, x^2]
    x_train = np.concatenate([x_train, x_train ** 2], axis=1)
    x_test = np.concatenate([x_test, x_test ** 2], axis=1)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    model_params = get_params(x_train.shape[1], y_train.shape[1])

    y_pred = forward(model_params, x_train)
    loss = loss_fn(y_pred, y_train)
    print(f'Initial Train Loss: {loss:.4f}')

    start = time.time()
    for epoch in range(1, epochs + 1):
        model_params = update(model_params, x_train, y_train, lr)
        if epoch % 100 == 0:
            y_pred = forward(model_params, x_train)
            loss = loss_fn(y_pred, y_train)
            print(f'Epoch {epoch}, Train Loss: {loss:.4f}')
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')

    y_pred = forward(model_params, x_test)
    loss = loss_fn(y_pred, y_test)
    print(f'Test Loss: {loss:.4f}')

    plt.scatter(x_test[:, :1], y_test, c='red', label='Test Data')
    plt.scatter(x_test[:, :1], y_pred, c='blue', label='Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main(**vars(get_parser(__doc__.split('\n')[1], lr=0.3, epochs=500)
                .parse_args()))
