# Tiny Reverse-Mode Autodiff in Python

A compact, educational implementation of **scalar reverse-mode automatic differentiation**, inspired by Ben Clarkson’s excellent blog post:

https://bclarkson-code.com/posts/llm-from-scratch-scalar-autograd/post.html

This version is a simplified, clean Python 3.12 implementation, with code adapted and rewritten to match my own learning goals.

It demonstrates the core ideas behind frameworks like PyTorch and JAX in just a few dozen lines of code.

See the full technical report explaining reverse-mode autodiff:  [report.md](./report.md)


## Features

* Scalar `Tensor` class (each node holds a single float)
* Reverse-mode autodiff with an explicit computation DAG
* Depth-first topological graph construction
* Derivative accumulation via the chain rule
* Operator overloading for `+`, `-`, `*`
* Mixes naturally with Python `int` / `float`
* `clear_derivatives()` to reset gradients between backward passes
* Simple, readable implementation suitable for learning or teaching

## Example

```python
from tensor import Tensor

x = Tensor(5, name="x")
c = Tensor(3, name="c")

y = 5 + x * (x + c)    # y = x^2 + cx + 5

y.backward()

print("y:", y.value)           # 45.0
print("dy/dx:", x.derivative)  # 2*5 + 3 = 13.0
print("dy/dc:", c.derivative)  # 5.0
```

A standalone version of this example is included in quicktest.py for convenience.

## Running Tests

Tests are located in `test_basic.py`.

### Install pytest

```bash
pip install pytest
```

### Run the test suite

```bash
pytest -q
```

Expected output:

```
......                                                                   [100%]
6 passed in 0.00s
```

## Project Structure

```
tensor.py                # core autodiff engine
test_basic.py            # pytest tests
quicktest.py             # simple runnable example
autodiff_sandbox.ipynb   # development/exploration notebook
README.md
```

## Notes

This project is intentionally small and focused.
It is meant to be a clear, approachable reference for understanding the mechanics of reverse-mode autodiff, without the complexity of full frameworks.