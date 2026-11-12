"""
version4.py: a multi-layer perceptron, no hand-made features needed.

What this file teaches
- Instead of handing the model the feature x^2, we give it the raw x and let
  hidden layers with a ReLU non-linearity learn the curve: 1 -> 64 -> 64 -> 1.
- The parameters are now a LIST of per-layer dicts. Because jax.grad and
  tree_map work on any pytree, the update step does not change at all.
- jax.random.split derives one independent key per layer from a single seed.

New compared to version3.py: the MLP (parameter list, forward loop with ReLU)
and raw inputs instead of [x, x^2].
"""
import time

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import generate_data, get_parser, set_seed, train_test_split


def get_mlp_params(param_list, seed=2025):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(param_list) - 1)
    params = []
    for i, (fan_in, fan_out) in enumerate(zip(param_list[:-1], param_list[1:])):
        w = jax.random.normal(keys[i], (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((fan_out,))
        params.append(dict(weights=w, biases=b))
    return params


@jax.jit
def forward(params, x):
    *hidden_layers, final_layer = params
    for layer in hidden_layers:
        x = jnp.dot(x, layer['weights']) + layer['biases']
        x = jax.nn.relu(x)
    x = jnp.dot(x, final_layer['weights']) + final_layer['biases']
    return x


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
    param_list = kwargs['param_list']

    set_seed(seed)

    x, y = generate_data()
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    model_params = get_mlp_params(param_list, seed)

    loss = loss_fn(model_params, x_train, y_train)
    print(f'Initial Train Loss: {loss:.4f}')

    start = time.time()
    num_batches = x_train.shape[0] // batch_size + int(
        x_train.shape[0] % batch_size > 0)
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

    plt.scatter(x_test, y_test, c='red', label='Test Data')
    y_pred = batched_forward(model_params, x_test)
    plt.scatter(x_test, y_pred, c='blue', label='Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    args = get_parser(__doc__.split('\n')[1], lr=0.01, epochs=500,
                      batch_size=50).parse_args()
    main(param_list=[1, 64, 64, 1], **vars(args))
