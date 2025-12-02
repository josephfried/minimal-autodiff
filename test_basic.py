from tensor import Tensor  # or whatever your module is called


def almost_equal(a, b, eps=1e-6):
    return abs(a - b) < eps


def test_square():
    x = Tensor(3, name="x")
    y = x * x              # y = x^2
    y.backward()
    assert almost_equal(x.derivative, 2 * 3)


def test_linear():
    m = Tensor(2, name="m")
    x = Tensor(4, name="x")
    c = Tensor(1, name="c")
    y = m * x + c          # y = 2x + 1
    y.backward()
    assert almost_equal(m.derivative, x.value)
    assert almost_equal(x.derivative, m.value)
    assert almost_equal(c.derivative, 1.0)


def test_reuse():
    x = Tensor(5, name="x")
    c = Tensor(3, name="c")
    y = (x + c) * x        # y = x^2 + cx → dy/dx = 2x + c, dy/dc = x
    y.backward()
    assert almost_equal(x.derivative, 2 * x.value + c.value)
    assert almost_equal(c.derivative, x.value)


def test_leaf_backward():
    # Calling backward on a leaf should give derivative 1.0 (f(x) = x)
    x = Tensor(7, name="x")
    x.backward()
    assert almost_equal(x.derivative, 1.0)


def test_python_scalars_left_and_right():
    x = Tensor(3, name="x")

    y1 = x + 2       # Tensor + int
    y2 = 2 + x       # int + Tensor
    y3 = x * 4       # Tensor * int
    y4 = 4 * x       # int * Tensor

    # Just check the forward values to make sure operator overloading works
    assert almost_equal(y1.value, 5.0)
    assert almost_equal(y2.value, 5.0)
    assert almost_equal(y3.value, 12.0)
    assert almost_equal(y4.value, 12.0)


def test_clear_derivatives_and_reuse_across_graphs():
    # Same Tensor used in two different graphs; clear_derivatives between them
    x = Tensor(2, name="x")
    c = Tensor(3, name="c")

    # First graph: y1 = (x + c) * x
    y1 = (x + c) * x
    y1.backward()
    assert almost_equal(x.derivative, 2 * x.value + c.value)
    assert almost_equal(c.derivative, x.value)

    # Clear derivatives on graph 2’s root (or on x,c directly, depending on your API)
    # Assuming you implemented clear_derivatives() as a method on the root:
    y2 = x * c
    y2.clear_derivatives()   # or x.clear_derivatives(); c.clear_derivatives()
    y2.backward()

    # For y2 = x * c: dy2/dx = c, dy2/dc = x
    assert almost_equal(x.derivative, c.value)
    assert almost_equal(c.derivative, x.value)
