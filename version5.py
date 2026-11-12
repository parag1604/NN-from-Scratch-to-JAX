"""
version5.py: classifying MNIST digits with the MLP and a softmax cross-entropy.

What this file teaches
- Same MLP as version4, now 784 -> 256 -> 32 -> 10: one output (a logit) per
  digit class.
- The softmax cross-entropy loss, written out: log_softmax(z) = z - logsumexp(z)
  (logsumexp is the numerically stable way to compute log sum_k e^{z_k}), and
  the loss is -mean(log p[true class]), using one-hot labels to pick it out.
- jax.value_and_grad(..., has_aux=True) returns the loss, an auxiliary output
  (the logits, reused for the accuracy) and the gradients in one pass.
- A train / evaluate split of the loop: evaluation does no update.
- The data is the Kaggle "MNIST in CSV" file, loaded with numpy; we hold out
  1/6 of it (10 000 images) as a test set.

New compared to version4.py: a real dataset, classification with softmax
cross-entropy, value_and_grad with auxiliary outputs, accuracy, and a plot
of predictions.
"""
import jax
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import get_mnist_data, get_parser, train_test_split, set_seed


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
    logits = jnp.dot(x, final_layer['weights']) + final_layer['biases']
    return logits


batched_forward = jax.vmap(forward, in_axes=(None, 0))


@jax.jit
def loss_fn(params, x, y):
    y_ohe = jax.nn.one_hot(y, num_classes=params[-1]['weights'].shape[1])
    logits = batched_forward(params, x)
    # log softmax, computed stably: z - log sum_k exp(z_k)
    log_probs =\
        logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss = -(log_probs * y_ohe).sum(axis=-1).mean()
    return loss, logits


@jax.jit
def update(params, x, y, lr=0.01):
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
    return jax.tree_util.tree_map(
        lambda p, g: p - lr * g, params, grads), loss, logits


def train(params, x, y, batch_size, lr=0.01, epoch=0):
    losses, corrects = 0, 0
    num_batches = len(x) // batch_size + int(len(x) % batch_size > 0)
    shuffled_idxs = np.random.permutation(x.shape[0])
    for i in range(num_batches):
        batch_idxs = shuffled_idxs[i * batch_size: i * batch_size + batch_size]
        params, loss, logits = update(params, x[batch_idxs], y[batch_idxs], lr)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y[batch_idxs])
        losses += loss * len(batch_idxs)
        corrects += acc * len(batch_idxs)
    loss = losses / len(x)
    acc = corrects / len(x)
    print(f'Epoch {epoch} - Train | accuracy: {acc:.4f} | loss: {loss:.4f}')
    return params


def evaluate(params, x, y, batch_size, epoch=-1):
    num_batches = len(x) // batch_size + int(len(x) % batch_size > 0)
    corrects, losses = 0, 0
    idxs = np.arange(len(x))
    for i in range(num_batches):
        batch_idxs = idxs[i * batch_size: (i + 1) * batch_size]
        loss, logits = loss_fn(params, x[batch_idxs], y[batch_idxs])
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y[batch_idxs])
        corrects += acc * len(batch_idxs)
        losses += loss * len(batch_idxs)
    loss = losses / len(x)
    acc = corrects / len(x)
    print(f'Epoch {epoch} -  Eval | accuracy: {acc:.4f} | loss: {loss:.4f}')


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    param_list = kwargs['param_list']

    set_seed(seed)

    x, y = get_mnist_data(kwargs['data'])
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=1 / 6)
    print(f'Train data shape: {x_train.shape}, {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, {y_test.shape}')

    model_params = get_mlp_params(param_list, seed)

    evaluate(model_params, x_test, y_test, batch_size, epoch=0)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model_params =\
            train(model_params, x_train, y_train, batch_size, lr, epoch)
        evaluate(model_params, x_test, y_test, batch_size, epoch)
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')

    # Visualize some predictions
    num_samples = 12  # 3 * 4 grid
    sample_idxs = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[sample_idxs].reshape(-1, 28, 28)
    sample_labels = y_test[sample_idxs]
    sample_preds = batched_forward(model_params, x_test[sample_idxs]).argmax(axis=-1)
    fig, axes = plt.subplots(3, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i], cmap='gray')
        ax.set_title(f'True: {sample_labels[i]}, Pred: {sample_preds[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = get_parser(__doc__.split('\n')[1], lr=0.1, epochs=40,
                        batch_size=512)
    parser.add_argument('--data', default='data/mnist_train.csv',
                        help='path to the Kaggle MNIST-in-CSV training file')
    main(param_list=[784, 256, 32, 10], **vars(parser.parse_args()))
