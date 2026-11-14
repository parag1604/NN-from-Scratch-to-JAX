# Neural Networks from Scratch to JAX

Code that starts from linear regression in numpy and rewrites it step by step: JAX (`jit`, `vmap`, `grad`), mini-batches, an MLP, MNIST classification, a proper data pipeline, and finally Flax NNX and Optax with a CNN. It is the JAX counterpart of [NN-from-Scratch-to-PyTorch](https://github.com/parag1604/NN-from-Scratch-to-PyTorch).

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Neural-Networks)

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy and Matplotlib
- JAX - install the accelerator build for your hardware if you have one (see https://docs.jax.dev/en/latest/installation.html)
- Flax (>= 0.11, which changed the `nnx.Optimizer` API) and Optax - versions 7 and 8
- PyTorch and Torchvision - only to download MNIST and batch it with a `DataLoader` (versions 6-8)

## Data

- Versions 0-4: synthetic data (`generate_data` in `utils.py`).
- Version 5: the Kaggle "MNIST in CSV" dataset, https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. Download it and place `mnist_train.csv` at `./data/mnist_train.csv`, or pass another path with `--data`. Only the training file is used: 1/6 of it is held out as a test set.
- Versions 6-8: the MNIST dataset (LeCun et al., 1998), http://yann.lecun.com/exdb/mnist/. It is downloaded automatically by torchvision into `./data/MNIST/` on the first run (change it with `--data-dir`).

## Running

Each file adds one idea. Every script accepts `--lr`, `--epochs` and `--seed` (and `--batch-size` from version 3 on), so `--epochs 1` gives a quick run.

### JAX basics (notebook)

`jax_basics.ipynb` is a short tour of device arrays, `jit`, `grad`, `vmap`, Jacobians and Hessians, jaxprs, and two gotchas (`nan` instead of errors; out-of-bounds indexing does not raise). Read it before version 1.

### 0. Linear regression in numpy

```bash
python version0.py
```

Fits y = 5 - 10x² - 3x + noise with a linear model on the features [x, x²]. Gradients are derived by hand, and training uses full-batch gradient descent. The noise has unit variance, so the best possible test loss (½ MSE) is about 0.5. With the default seed the test loss is 0.51.

### 1. The same model in JAX: jit, vmap, device_put

```bash
python version1.py
```

`jax.numpy` replaces numpy, `@jax.jit` compiles, `jax.vmap` batches a single-example `forward`, and PRNG keys are explicit. On such a tiny problem the run is slower than numpy, because the time includes compilation (see version 5b).

### 2. Automatic differentiation with jax.grad

```bash
python version2.py
```

`jax.grad(loss_fn)` replaces the hand-derived gradients, and `jax.tree_util.tree_map` does the update for any pytree of parameters. The results are the same as version 1.

### 3. Mini-batch SGD

```bash
python version3.py --batch-size 50
```

Shuffles each epoch and updates on mini-batches.

### 4. A multi-layer perceptron

```bash
python version4.py
```

An MLP (1-64-64-1, ReLU) on the raw x learns the curve without the hand-made x² feature. The parameters are a list of per-layer dicts, and the update code is unchanged.

### 5. MNIST classification with softmax cross-entropy

```bash
python version5.py --data data/mnist_train.csv
```

A 784-256-32-10 MLP is trained with log-softmax (via `logsumexp`) and cross-entropy written by hand. `jax.value_and_grad(..., has_aux=True)` returns the loss, the logits and the gradients in one pass. The script prints the train and test accuracy each epoch and plots 12 test digits with their predictions.

### 5b. (Optional) What does jit buy us?

```bash
python version5b.py
```

Times the forward pass of the version 5 MLP shape in numpy and in jitted JAX on random inputs: the first call (compilation), then the compiled calls. The numbers depend on your hardware.

### 6. A data pipeline: torchvision, DataLoader and a validation split

```bash
python version6.py
```

The same model, now fed by torchvision MNIST through a PyTorch `DataLoader` with a numpy `collate_fn` (`get_mnist_loaders` in `utils.py`). It uses train/validation/test splits of 50k/10k/10k.

### 7. Flax NNX and Optax

```bash
python version7.py
```

The MLP becomes an `nnx.Module` built from `nnx.Linear` layers. Optax provides SGD (try `optax.adam`) and the cross-entropy loss, and `nnx.MultiMetric` tracks accuracy and loss.

### 8. A CNN with dropout

```bash
python version8.py
```

Two convolution + average-pooling blocks, dropout and three dense layers. The script switches between `model.train()` and `model.eval()` so that dropout is only active during training. Images are fed as (28, 28, 1) instead of being flattened.

## Related

Suggested order for the neural-network repositories:

1. [Perceptron](https://github.com/parag1604/Perceptron): the perceptron learning rule, its convergence, and where it fails (XOR)
2. [MLP-Backprop-in-NumPy](https://github.com/parag1604/MLP-Backprop-in-NumPy): a two-layer network in numpy with hand-written backpropagation, gradient checking and optimizers
3. [Scalar-Autograd](https://github.com/parag1604/Scalar-Autograd): a tiny scalar automatic-differentiation engine that automates backpropagation
4. [NN-from-Scratch-to-PyTorch](https://github.com/parag1604/NN-from-Scratch-to-PyTorch): from a numpy MLP to PyTorch autograd, layers and optimizers, ending on MNIST
5. [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): CNNs in PyTorch on MNIST, Fashion-MNIST and CIFAR-10, with batch norm and dropout
6. **NN-from-Scratch-to-JAX** (this repository): from numpy linear regression to JAX, Flax and Optax, ending with a CNN on MNIST
