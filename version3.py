"""
version3.py: mini-batch stochastic gradient descent.

What this file teaches
- Full-batch gradient descent touches every example for each update, which
  does not scale to large datasets. Mini-batch SGD estimates the gradient from
  a small random batch: many cheap, noisy updates per epoch instead of one
  exact one.
- Each epoch we shuffle the training indices and walk through them in chunks
  of batch_size (the last batch may be smaller).
- Noisier gradients call for a smaller learning rate (0.03 here vs 0.3).
- Note: jit compiles once per input SHAPE, so a smaller last batch triggers
  one extra compilation (dropping it, or padding, avoids that).

New compared to version2.py: the mini-batch loop (the model and the update
are unchanged).
"""
import time

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import generate_data, get_parser, set_seed, train_test_split


def get_params(fan_in, fan_out, seed=2025):
    key = jax.random.PRNGKey(seed)
    return dict(
        weights=jax.random.normal(key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in),
        bias=jnp.zeros((fan_out,)),
    )


@jax.jit
def forward(params, x):
    return jnp.dot(x, params['weights']) + params['bias']


batched_forward = jax.vmap(forward, in_axes=(None, 0))


@jax.jit
def loss_fn(params, x, y):
    return 0.5 * jnp.mean((batched_forward(params, x) - y) ** 2)


@jax.jit
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)

    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']

    set_seed(seed)

    x, y = generate_data()
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    x_train = np.concatenate([x_train, x_train ** 2], axis=1)
    x_test = np.concatenate([x_test, x_test ** 2], axis=1)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    model_params = get_params(x_train.shape[1], y_train.shape[1], seed)

    loss = loss_fn(model_params, x_train, y_train)
    print(f'Initial Train Loss: {loss:.4f}')

    start = time.time()
    num_batches = x_train.shape[0] // batch_size + int(x_train.shape[0] % batch_size > 0)
    for epoch in range(1, epochs + 1):
        shuffled_idxs = np.random.permutation(x_train.shape[0])
        for i in range(num_batches):
            idxs = shuffled_idxs[i * batch_size: i * batch_size + batch_size]
            x, y = x_train[idxs], y_train[idxs]
            model_params = update(model_params, x, y, lr)
        if epoch % 100 == 0:
            loss = loss_fn(model_params, x_train, y_train)
            print(f'Epoch {epoch}, Train Loss: {loss:.4f}')
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')

    loss = loss_fn(model_params, x_test, y_test)
    print(f'Test Loss: {loss:.4f}')

    plt.scatter(x_test[:, :1], y_test, c='red', label='Test Data')
    y_pred = batched_forward(model_params, x_test)
    plt.scatter(x_test[:, :1], y_pred, c='blue', label='Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main(**vars(get_parser(__doc__.split('\n')[1], lr=0.03, epochs=500,
                           batch_size=50).parse_args()))
