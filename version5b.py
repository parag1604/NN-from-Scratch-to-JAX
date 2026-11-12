"""
version5b.py (optional side quest): how much does jax.jit actually buy us?

What this file teaches
- We time the forward pass of version5's MLP shape (784 -> 256 -> 32 -> 10, on
  a batch of 512 random inputs) two ways: plain numpy and JAX with @jit.
- The first call of a jitted function pays for tracing + XLA compilation;
  later calls with the same shapes reuse the compiled code.
- JAX dispatches work asynchronously: a call returns before the result is
  computed. For honest timings call .block_until_ready() on the output.
- No data is needed: the inputs and weights are random.

New compared to version5.py: nothing is added to the model; this is a
benchmark that explains why every function in version1-5 is decorated with
@jax.jit. Results depend heavily on your hardware (CPU vs GPU).
"""
import time

import numpy as np
import jax.numpy as jnp
from jax import jit, device_put


def calc(t):
    """Pretty-print a duration in s / ms / µs."""
    unit = "s"
    if t < 1:
        unit, t = "ms", t * 1000
        if t < 1:
            unit, t = "µs", t * 1000
    return f"{t:.2f} {unit}"


def timeit(func, is_jax, *args, repeats=10):
    total = 0.0
    minimum = float("inf")
    for _ in range(repeats):
        start = time.time()
        if is_jax:
            func(*args).block_until_ready()  # wait for the async computation
        else:
            func(*args)
        duration = time.time() - start
        minimum = min(minimum, duration)
        total += duration
    print(f"Average time: {calc(total / repeats)} | Minimum time: {calc(minimum)}")


def func1(x, w1, w2, w3):
    x = np.dot(x, w1)
    x = np.tanh(x)
    x = np.dot(x, w2)
    x = np.tanh(x)
    x = np.dot(x, w3)
    return x


@jit
def func2(x, w1, w2, w3):
    x = jnp.dot(x, w1)
    x = jnp.tanh(x)
    x = jnp.dot(x, w2)
    x = jnp.tanh(x)
    x = jnp.dot(x, w3)
    return x


def main():
    # float32 for both, so the comparison is fair (numpy defaults to float64)
    w1 = np.random.randn(784, 256).astype(np.float32)
    w2 = np.random.randn(256, 32).astype(np.float32)
    w3 = np.random.randn(32, 10).astype(np.float32)
    x = np.random.randn(512, 784).astype(np.float32)

    jw1, jw2, jw3, jx = (device_put(a) for a in (w1, w2, w3, x))

    print("Numpy Inference:", end=" ")
    timeit(func1, False, x, w1, w2, w3)

    # the first call traces and compiles the JAX function
    start = time.time()
    func2(jx, jw1, jw2, jw3).block_until_ready()
    print(f"JAX Compilation (first call): {calc(time.time() - start)}")

    print("JAX Inference:", end=" ")
    timeit(func2, True, jx, jw1, jw2, jw3)


if __name__ == '__main__':
    main()
