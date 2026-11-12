"""
version2.py: automatic differentiation with jax.grad.

What this file teaches
- jax.grad(f) returns a new function that computes the gradient of f with
  respect to its FIRST argument. Here that argument is the whole params dict,
  so the gradient comes back as a dict with the same structure.
- For that, the loss must be a function of the parameters: loss_fn now takes
  (params, x, y) and runs the forward pass itself.
- Nested containers of arrays (dicts, lists, tuples) are "pytrees", and
  jax.tree_util.tree_map applies a function leaf by leaf, so the gradient
  descent step p <- p - lr * g is one line for any model.

New compared to version1.py: jax.grad replaces the hand-derived gradients,
and tree_map replaces the per-parameter updates.
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
    grads = jax.grad(loss_fn)(params, x, y)  # same structure as params

    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']

    set_seed(seed)

    x, y = generate_data()
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    x_train = np.concatenate([x_train, x_train ** 2], axis=1)
    x_test = np.concatenate([x_test, x_test ** 2], axis=1)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    # load data to device
    x_train = jax.device_put(x_train)
    y_train = jax.device_put(y_train)
    x_test = jax.device_put(x_test)
    y_test = jax.device_put(y_test)

    model_params = get_params(x_train.shape[1], y_train.shape[1], seed)

    loss = loss_fn(model_params, x_train, y_train)
    print(f'Initial Train Loss: {loss:.4f}')

    start = time.time()
    for epoch in range(1, epochs + 1):
        model_params = update(model_params, x_train, y_train, lr)
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
    main(**vars(get_parser(__doc__.split('\n')[1], lr=0.3, epochs=500)
                .parse_args()))
