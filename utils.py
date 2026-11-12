"""
Shared helpers: seeding, the toy regression data, and the two MNIST loaders
(the Kaggle CSV used by version5, and torchvision used from version6 on).
"""
import argparse

import numpy as np


def set_seed(seed=2025):
    np.random.seed(seed)
    try:  # torch is only needed from version6 on (for its DataLoader)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_parser(description, **defaults):
    """Command line flags shared by every version (defaults per version)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 0.3))
    parser.add_argument('--epochs', type=int,
                        default=defaults.get('epochs', 500))
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 2025))
    if 'batch_size' in defaults:
        parser.add_argument('--batch-size', type=int,
                            default=defaults['batch_size'])
    return parser


def generate_data(num_samples=200):
    """Noisy samples of the quadratic y = 5 - 10 x^2 - 3 x on [-1, 1]."""
    x = np.random.uniform(-1, 1, (num_samples, 1))
    noise = np.random.normal(0, 1, (num_samples, 1))
    y = 5 - 10 * (x ** 2) - 3 * x + noise
    return x, y


def get_mnist_data(path="data/mnist_train.csv"):
    """
    Loads the Kaggle "MNIST in CSV" training file: a header row, then one row
    per image with the label followed by 784 pixel values in [0, 255].
    """
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)
    x = data[:, 1:] / 255.0  # Normalize pixel values to [0, 1]
    y = data[:, 0].astype(int)  # Labels
    print(f'Loaded {path}: x {x.shape}, y {y.shape}')
    return x, y


def train_test_split(x, y, test_size=0.25):
    num_samples = len(x)
    indices = np.random.permutation(num_samples)
    split_index = int(num_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def collate_np(batch):
    """Turn a list of (image, label) pairs into numpy arrays JAX can consume."""
    x, y = zip(*batch)
    return np.array(x) / 255., np.array(y)


def get_mnist_loaders(batch_size, flatten=True, root="data", seed=2025,
                      val_fraction=1 / 6):
    """
    torchvision MNIST (downloaded to `root` on first use) split into
    train / validation / test DataLoaders that yield numpy arrays.
    flatten=True gives (B, 784) vectors for MLPs, flatten=False gives
    (B, 28, 28, 1) images (channels last, as Flax convolutions expect).
    """
    from torch import Generator
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets

    if flatten:
        transform = lambda x: np.ravel(np.array(x, dtype=np.float32))
    else:
        transform = lambda x: np.array(x, dtype=np.float32)[..., np.newaxis]

    train_data = datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(
        root=root, train=False, download=True, transform=transform)
    # hold out part of the training set for validation (50k / 10k for MNIST)
    num_val = int(len(train_data) * val_fraction)
    train_data, val_data = random_split(
        train_data, [len(train_data) - num_val, num_val],
        generator=Generator().manual_seed(seed))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_np)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_np)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_np)
    return train_loader, val_loader, test_loader
