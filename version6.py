"""
version6.py: a proper input pipeline, torchvision + DataLoader + validation set.

What this file teaches
- JAX has no data-loading library of its own, and does not need one: any
  iterator that yields numpy arrays works. Here torchvision downloads MNIST and
  a PyTorch DataLoader batches and shuffles it; a custom collate_fn returns
  numpy arrays instead of torch tensors (see get_mnist_loaders in utils.py).
- Three splits: train (50k) to fit the weights, validation (10k, carved out of
  the official training set) to monitor training and choose hyper-parameters,
  and the official test set (10k), touched once at the very end.

New compared to version5.py: the torchvision/DataLoader pipeline and the
train/validation/test protocol. The model, loss and update are unchanged.
"""
import jax
import time
import jax.numpy as jnp

from utils import get_mnist_loaders, get_parser, set_seed


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
    log_probs =\
        logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss = -(log_probs * y_ohe).sum(axis=-1).mean()
    return loss, logits


@jax.jit
def update(params, x, y, lr=0.01):
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
    return jax.tree_util.tree_map(
        lambda p, g: p - lr * g, params, grads), loss, logits


def train(params, dataloader, lr=0.01, epoch=0):
    losses, corrects, counts = 0, 0, 0
    for x, y in dataloader:
        params, loss, logits = update(params, x, y, lr)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        losses += loss * len(x)
        corrects += acc * len(x)
        counts += len(x)
    loss = losses / counts
    acc = corrects / counts
    print(f'Epoch {epoch} - Train | accuracy: {acc:.4f} | loss: {loss:.4f}')
    return params


def evaluate(params, dataloader, epoch=-1):
    corrects, losses, counts = 0, 0, 0
    for x, y in dataloader:
        loss, logits = loss_fn(params, x, y)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        corrects += acc * len(x)
        losses += loss * len(x)
        counts += len(x)
    loss = losses / counts
    acc = corrects / counts
    if epoch == -1:
        print(f'Final    -  Test | accuracy: {acc:.4f} | loss: {loss:.4f}')
    else:
        print(f'Epoch {epoch} -  Eval | accuracy: {acc:.4f} | loss: {loss:.4f}')


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    param_list = kwargs['param_list']

    set_seed(seed)

    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size, flatten=True, root=kwargs['data_dir'], seed=seed)

    model_params = get_mlp_params(param_list, seed=seed)

    evaluate(model_params, val_loader, 0)
    start = time.time()
    for epoch in range(1, epochs + 1):
        model_params = train(model_params, train_loader, lr, epoch)
        evaluate(model_params, val_loader, epoch)
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')
    evaluate(model_params, test_loader)


if __name__ == '__main__':
    parser = get_parser(__doc__.split('\n')[1], lr=0.1, epochs=40,
                        batch_size=512)
    parser.add_argument('--data-dir', default='data',
                        help='where torchvision stores/downloads MNIST')
    main(param_list=[784, 256, 32, 10], **vars(parser.parse_args()))
