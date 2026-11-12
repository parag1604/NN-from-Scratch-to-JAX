"""
version7.py: the same MLP with Flax (NNX) layers and an Optax optimizer.

What this file teaches
- Flax NNX lets us write the network as a Python class, much like
  torch.nn.Module: nnx.Linear layers are created in __init__ and called in
  __call__. Parameters live inside the module, so no more lists of dicts.
  Randomness for initialisation comes from an explicit nnx.Rngs(seed).
- Optax provides the optimizers. nnx.Optimizer(model, optax.sgd(lr),
  wrt=nnx.Param) holds the optimizer state, and optimizer.update(model, grads)
  applies one step to the model's parameters in place.
  (Swap optax.sgd for optax.adam to try another optimizer.)
- optax.softmax_cross_entropy_with_integer_labels replaces our hand-written
  log-softmax and one-hot code from version5.
- nnx.MultiMetric accumulates accuracy and average loss across batches.
- nnx.jit / nnx.value_and_grad are the Flax-aware versions of jax.jit /
  jax.value_and_grad that understand stateful modules.

New compared to version6.py: Flax NNX for the model, Optax for the update,
and nnx metrics. The data pipeline is unchanged.
Requires flax >= 0.11 (the Optimizer API changed in 0.11).
"""
import time

import optax
from flax import nnx

from utils import get_mnist_loaders, get_parser, set_seed


class Network(nnx.Module):
    def __init__(self, param_list, rngs):
        self.fc1 = nnx.Linear(param_list[0], param_list[1], rngs=rngs)
        self.fc2 = nnx.Linear(param_list[1], param_list[2], rngs=rngs)
        self.fc3 = nnx.Linear(param_list[2], param_list[3], rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)  # logits


def loss_fn(model, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y).mean()
    return loss, logits


@nnx.jit
def train_step(model, optimizer, metrics, x, y):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, x, y)
    metrics.update(loss=loss, logits=logits, labels=y)
    optimizer.update(model, grads)  # in-place update of the model's params


@nnx.jit
def eval_step(model, metrics, x, y):
    loss, logits = loss_fn(model, x, y)
    metrics.update(loss=loss, logits=logits, labels=y)


def train(dataloader, model, optimizer, metrics, epoch):
    for x, y in dataloader:
        train_step(model, optimizer, metrics, x, y)
    print(f'Epoch {epoch} - Train', end=' ')
    for metric, value in metrics.compute().items():
        print(f'| {metric}: {value:.4f}', end=' ')
    print()
    metrics.reset()


def evaluate(dataloader, model, metrics, epoch=-1):
    for x, y in dataloader:
        eval_step(model, metrics, x, y)
    if epoch == -1:
        print('Final    -  Test', end=' ')
    else:
        print(f'Epoch {epoch} -  Eval', end=' ')
    for metric, value in metrics.compute().items():
        print(f'| {metric}: {value:.4f}', end=' ')
    print()
    metrics.reset()


def main(**kwargs):
    lr = kwargs['lr']
    seed = kwargs['seed']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    param_list = kwargs['param_list']

    set_seed(seed)

    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size, flatten=True, root=kwargs['data_dir'], seed=seed)

    model = Network(param_list, nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))

    evaluate(val_loader, model, metrics, epoch=0)
    start = time.time()
    for epoch in range(1, epochs + 1):
        train(train_loader, model, optimizer, metrics, epoch)
        evaluate(val_loader, model, metrics, epoch)
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')
    evaluate(test_loader, model, metrics)


if __name__ == '__main__':
    parser = get_parser(__doc__.split('\n')[1], lr=0.1, epochs=40,
                        batch_size=512)
    parser.add_argument('--data-dir', default='data',
                        help='where torchvision stores/downloads MNIST')
    main(param_list=[784, 256, 32, 10], **vars(parser.parse_args()))
