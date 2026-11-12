"""
version1.py: the same model in JAX: jax.numpy, jit, vmap and devices.

What this file teaches
- jax.numpy mirrors the numpy API, so the model code barely changes.
  (Read jax_basics.ipynb first for a tour of jit / grad / vmap.)
- @jax.jit traces a function once, compiles it with XLA and reuses the
  compiled version on later calls with the same shapes: the first call is
  slow, the rest are fast. Jitted functions must be pure (no side effects).
- jax.vmap turns a function written for ONE example into one that works on a
  batch: forward() below handles a single x of shape (2,), and
  batched_forward maps it over the leading axis of a (N, 2) array.
- JAX arrays are immutable. `params['weights'] -= ...` does not modify an
  array in place, it builds a new array and rebinds the dict entry.
- jax.device_put moves data to the accelerator (GPU/TPU if present) once,
  instead of copying it on every call.
- Randomness is explicit: jax.random.PRNGKey(seed) instead of a global seed.

New compared to version0.py: JAX instead of numpy, jit, vmap, device_put,
explicit PRNG keys, and He initialisation (std = sqrt(2 / fan_in)).
The gradients are still derived by hand.
"""
import time

import jax
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
def loss_fn(y_pred, y_true):
    return 0.5 * jnp.mean((y_pred - y_true) ** 2)


@jax.jit
def forward(params, x):
    return jnp.dot(x, params['weights']) + params['bias']


# forward() is written for a single example; vmap maps it over the batch axis
# of x (in_axes=0) while sharing the same params (in_axes=None)
batched_forward = jax.vmap(forward, in_axes=(None, 0))


@jax.jit
def update(params, x, y, lr=0.01):
    y_pred = batched_forward(params, x)

    grad_weights = jnp.dot(x.T, (y_pred - y)) / x.shape[0]
    grad_bias = jnp.mean(y_pred - y, axis=0)

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
    x_train = jnp.concatenate([x_train, x_train ** 2], axis=1)
    x_test = jnp.concatenate([x_test, x_test ** 2], axis=1)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    # load data to device
    x_train = jax.device_put(x_train)
    y_train = jax.device_put(y_train)
    x_test = jax.device_put(x_test)
    y_test = jax.device_put(y_test)

    model_params = get_params(x_train.shape[1], y_train.shape[1], seed)

    y_pred = batched_forward(model_params, x_train)
    loss = loss_fn(y_pred, y_train)
    print(f'Initial Train Loss: {loss:.4f}')

    start = time.time()
    for epoch in range(1, epochs + 1):
        model_params = update(model_params, x_train, y_train, lr)
        if epoch % 100 == 0:
            y_pred = batched_forward(model_params, x_train)
            loss = loss_fn(y_pred, y_train)
            print(f'Epoch {epoch}, Train Loss: {loss:.4f}')
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds (includes compilation)')

    y_pred = batched_forward(model_params, x_test)
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
