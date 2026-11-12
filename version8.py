"""
version8.py: a convolutional network (CNN) with dropout, in Flax NNX.

What this file teaches
- Images have spatial structure that an MLP throws away by flattening.
  Convolutions slide small learned filters over the image, sharing weights
  across positions: far fewer parameters, and translation equivariance.
- Architecture: conv(3x3, 32) -> ReLU -> 2x2 average pool -> conv(3x3, 64)
  -> ReLU -> 2x2 average pool -> flatten (7 * 7 * 64 = 3136) -> dropout
  -> 256 -> 32 -> 10. Flax convolutions expect channels-last input
  (batch, height, width, channels), so images are loaded as (28, 28, 1).
- Dropout randomly zeroes activations during training only, so the model has
  a training mode and an evaluation mode: model.train() / model.eval().

New compared to version7.py: nnx.Conv, pooling, nnx.Dropout and train/eval
modes; images are no longer flattened.
Requires flax >= 0.11.
"""
import time
from functools import partial

import optax
from flax import nnx

from utils import get_mnist_loaders, get_parser, set_seed


class Network(nnx.Module):
    def __init__(self, rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.fc1 = nnx.Linear(3136, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 32, rngs=rngs)
        self.fc3 = nnx.Linear(32, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))  # (B, 14, 14, 32)
        x = self.avg_pool(nnx.relu(self.conv2(x)))  # (B, 7, 7, 64)
        x = x.reshape(x.shape[0], -1)               # (B, 3136)
        x = self.dropout(x)  # active after model.train(), off after model.eval()
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)


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

    set_seed(seed)

    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size, flatten=False, root=kwargs['data_dir'], seed=seed)

    model = Network(nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))

    model.eval()
    evaluate(val_loader, model, metrics, epoch=0)
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()  # dropout on
        train(train_loader, model, optimizer, metrics, epoch)
        model.eval()   # dropout off
        evaluate(val_loader, model, metrics, epoch)
    end = time.time()
    print(f'Training time: {end - start:.4f} seconds')
    evaluate(test_loader, model, metrics)


if __name__ == '__main__':
    parser = get_parser(__doc__.split('\n')[1], lr=0.1, epochs=40,
                        batch_size=512)
    parser.add_argument('--data-dir', default='data',
                        help='where torchvision stores/downloads MNIST')
    main(**vars(parser.parse_args()))
