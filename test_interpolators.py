from function_interpolator import BoundedInterpolator, DynamicBoundedInterpolator
import numpy as np
from tictoc import *
import math


def assert_isclose(a, b, rel_tol=1e-9):
    if not math.isclose(a, b, rel_tol=rel_tol):
        raise AssertionError(f"{a} is not close to {b}")


def test_dynamic_1():
    f = lambda x: x


    i = DynamicBoundedInterpolator(f, 0, 1, 0.1)
    print(i(1))

    lo = (len(i._data) - 1) * i._dx + i._x1
    hi = len(i._data) * i._dx + i._x1

    print(lo, hi)
    print(i._x2)

    # print(i._data)



# x < x2
# int((x - x1) / dx) == len(data) - 1
# len(data) - 1 <= (x - x1) / dx < len(data)
# (len(data) - 1) * dx <= (x - x1) < len(data) * dx
# (len(data) - 1) * dx + x1 <= x < len(data) * dx + x1

def test_dynamic_2():
    f = lambda x: x + x * x

    a, b = 0, 10000
    res = 0.1
    step = 0.103
    expand_factor = 2.0
    rel_tol = 1e-1

    i1 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)

    tic(f'{a} to {b}, always expanding')
    for x in np.arange(a, b, step):
        assert_isclose(i1(x), f(x), rel_tol=rel_tol)
    tocl()

    i2 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)
    tic(f'{a} to {b}, already expanded')
    i2((a + b) / 2)
    for x in np.arange(a, b, step):
        assert_isclose(i2(x), f(x), rel_tol=rel_tol)
    tocl()

    i3 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)
    tic(f'{b} to {a}, always expanding')
    for x in np.arange(b, a, -step):
        assert_isclose(i3(x), f(x), rel_tol=rel_tol)
    tocl()

    i4 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)
    tic(f'{b} to {a}, already expanded')
    i4((a + b) / 2)
    for x in np.arange(b, a, -step):
        assert_isclose(i4(x), f(x), rel_tol=rel_tol)
    tocl()

    # print(i4._data[-10])
    # print(i3._data[-10])


if __name__ == '__main__':
    test_dynamic_1()
    test_dynamic_2()
