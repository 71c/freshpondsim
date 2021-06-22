from function_interpolator import DynamicBoundedInterpolator
import numpy as np
from tictoc import *
import math
import timeit



# have old version to compare speeds
from time import time
from collections import deque

class DynamicBoundedInterpolatorOld:
    def __init__(self, func, x_min, x_max, resolution, expand_factor=2.0, debug=False):
        self._func = func
        self._dx = float(resolution)
        self._expand_factor = expand_factor
        self._x1 = float(x_min)
        self._x2 = float(x_min)
        self._data = deque([self._func(self._x2)])
        self._debug = debug
        self._expand_right(x_max)

        # vectorized function so it can take ndarrays
        self._vf = np.vectorize(self._eval, otypes=[float])

    def _expand_right(self, x_max):
        """Expands the domain of the function to include at least x_max"""

        if self._debug:
            print(f"Expanding right, current x2: {self._x2}")
            t = time()

        n_new_samples = math.ceil((x_max - self._x2) / self._dx)
        self._data.extend([
            self._func(self._x2 + i * self._dx)
            for i in range(1, n_new_samples + 1)
        ])
        self._x2 = self._x2 + n_new_samples * self._dx

        if self._debug:
            print(f"new x2: {self._x2}, took {time() - t:.6f} seconds")

    def _expand_left(self, x_min):
        """Expands the domain of the function to include at least x_min"""

        if self._debug:
            print(f"Expanding left, current x1: {self._x1}")
            t = time()

        n_new_samples = math.ceil((self._x1 - x_min) / self._dx)
        self._data.extendleft([
            self._func(self._x1 - i * self._dx)
            for i in range(1, n_new_samples + 1)
        ])
        self._x1 = self._x1 - n_new_samples * self._dx

        if self._debug:
            print(f"new x1: {self._x1}, took {time() - t:.6f} seconds")

    def __call__(self, x):
        if type(x) is np.ndarray or type(x) is list:
            return self._vf(x)
        return self._eval(x)

    def _eval(self, x):
        if x < self._x1:
            diff = self._x1 - x
            threshold = (self._x2 - self._x1) * (self._expand_factor - 1)
            if diff <= threshold:
                new_x1 = self._x1 - threshold
            else:
                new_x1 = self._x1 - self._expand_factor * diff
            self._expand_left(new_x1)
        elif x > self._x2:
            diff = x - self._x2
            threshold = (self._x2 - self._x1) * (self._expand_factor - 1)
            if diff <= threshold:
                new_x2 = self._x2 + threshold
            else:
                new_x2 = self._x2 + self._expand_factor * diff
            self._expand_right(new_x2)

        pos = (x - self._x1) / self._dx
        k = int(pos)
        if k == pos:
            return self._data[k]
        elif k == len(self._data) - 1: # edge-case
            return self._data[k] # technically this is not accurate
        else:
            lval = self._data[k]
            rval = self._data[k + 1]
            ldiff = pos - k
            rdiff = 1 - ldiff
            return lval * rdiff + rval * ldiff


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


def test_dynamic_3():
    f = lambda x: x + x * x

    a, b = 0, 1000
    res = 0.1
    step = 0.103
    expand_factor = 2.0
    rel_tol = 1e-1

    def always_expanding_up():
        i1 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)
        for x in np.arange(a, b, step):
            assert_isclose(i1(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(always_expanding_up, number=100)
    print(f'{a} to {b}, always expanding: {t}')

    def already_expanded_up():
        i2 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)
        i2((a + b) / 2)
        for x in np.arange(a, b, step):
            assert_isclose(i2(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_up, number=100)
    print(f'{a} to {b}, already expanded: {t}')

    def always_expanding_down():
        i3 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)    
        for x in np.arange(b, a, -step):
            assert_isclose(i3(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(always_expanding_down, number=100)
    print(f'{b} to {a}, always expanding: {t}')

    def already_expanded_down():
        i4 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)
        i4((a + b) / 2)
        for x in np.arange(b, a, -step):
            assert_isclose(i4(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_down, number=100)
    print(f'{b} to {a}, already expanded: {t}')


def test_dynamic_3_compare():
    f = lambda x: x + x * x

    a, b = 0, 1000
    res = 0.1
    step = 0.103
    expand_factor = 2.0
    rel_tol = 1e-1

    def always_expanding_up():
        i1 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)
        for x in np.arange(a, b, step):
            assert_isclose(i1(x), f(x), rel_tol=rel_tol)
    
    tNO = timeit.timeit(always_expanding_up, number=100) # Dummy

    t = timeit.timeit(always_expanding_up, number=100)
    print(f'{a} to {b}, always expanding: {t}')

    def always_expanding_up_old():
        i1 = DynamicBoundedInterpolatorOld(f, a, a, res, expand_factor=expand_factor, debug=False)
        for x in np.arange(a, b, step):
            assert_isclose(i1(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(always_expanding_up_old, number=100)
    print(f'{a} to {b}, always expanding Old: {t}')


    def already_expanded_up():
        i2 = DynamicBoundedInterpolator(f, a, a, res, expand_factor=expand_factor, debug=False)
        i2((a + b) / 2)
        for x in np.arange(a, b, step):
            assert_isclose(i2(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_up, number=100)
    print(f'{a} to {b}, already expanded: {t}')

    def already_expanded_up_old():
        i2 = DynamicBoundedInterpolatorOld(f, a, a, res, expand_factor=expand_factor, debug=False)
        i2((a + b) / 2)
        for x in np.arange(a, b, step):
            assert_isclose(i2(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_up_old, number=100)
    print(f'{a} to {b}, already expanded Old: {t}')


    def always_expanding_down():
        i3 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)    
        for x in np.arange(b, a, -step):
            assert_isclose(i3(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(always_expanding_down, number=100)
    print(f'{b} to {a}, always expanding: {t}')

    def always_expanding_down_old():
        i3 = DynamicBoundedInterpolatorOld(f, b, b, res, expand_factor=expand_factor, debug=False)    
        for x in np.arange(b, a, -step):
            assert_isclose(i3(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(always_expanding_down_old, number=100)
    print(f'{b} to {a}, always expanding Old: {t}')


    def already_expanded_down():
        i4 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)
        i4((a + b) / 2)
        for x in np.arange(b, a, -step):
            assert_isclose(i4(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_down, number=100)
    print(f'{b} to {a}, already expanded: {t}')

    def already_expanded_down_old():
        i4 = DynamicBoundedInterpolatorOld(f, b, b, res, expand_factor=expand_factor, debug=False)
        i4((a + b) / 2)
        for x in np.arange(b, a, -step):
            assert_isclose(i4(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(already_expanded_down_old, number=100)
    print(f'{b} to {a}, already expanded Old: {t}')


def test_dynamic_4():
    f = lambda x: x + x * x

    a, b = 0, 1000
    res = 0.1
    expand_factor = 2.0
    rel_tol = 1e-1

    def test():
        i5 = DynamicBoundedInterpolator(f, b, b, res, expand_factor=expand_factor, debug=False)
        for x in np.logspace(0, 4, base=10, num=5):
            assert_isclose(i5(x), f(x), rel_tol=rel_tol)
    t = timeit.timeit(test, number=100)
    print(f'test: {t}')


def test_dynamic_5():
    f = lambda x: x

    a, b = 0, 1000
    res = 1
    expand_factor = 2
    rel_tol = 1e-1

    i = DynamicBoundedInterpolator(f, a, b, res, expand_factor=expand_factor, debug=False)
    ar = ([0, 1, 2, 3, 0.5, 0.9, 1.8])
    print(type(i(5.0)))
    print(type(i(ar)[0]))
    print(i([0.5, 0.8]))
    print(i._x1, i._x2)


def test_dynamic_6():
    # f = math.sqrt
    # i = DynamicBoundedInterpolator(f, 0, 3, 1, x_max=5, expand_factor=2, debug=False)
    # print(i(4))

    f = math.sqrt
    i = DynamicBoundedInterpolator(f, 2, 4, 2, x_min=0, x_max=6, expand_factor=2, debug=False)
    print(i(5.5), i._right_val)
    print(i._data)

    # print(i._x_min, i._x_max, i._x1, i._x2)
    # print(i(0.5))
    
    # print(i._x_min, i._x_max, i._x1, i._x2)
    # print(i._data)


def test_dynamic_extensive():
    f = lambda x: 2*x

    i = DynamicBoundedInterpolator(f,
        x1=1, x2=3, resolution=2, x_min=0, x_max=6, expand_factor=2)
    
    assert i._x1 == 1
    assert i._x2 == 3
    assert i._at_min == True
    assert i._at_max == False
    assert i._left_val == 0
    assert i._right_val == None

    assert i(0.9) == 1.8
    assert i._x1 == 1
    
    assert i(3.5) == 7
    assert i._x2 == 5
    assert i._at_max == False

    assert i(5.6) == 11.2
    assert i._x2 == 5
    assert i._at_max == True

    assert math.isclose(i(2.4), 4.8, rel_tol=1e-15)

    i = DynamicBoundedInterpolator(f,
        x1=1, x2=4, resolution=2, x_min=0.5, x_max=7, expand_factor=2)
    assert i._x1 == 1
    assert i._x2 == 5
    assert i._at_min == True
    assert i._at_max == False
    assert i._left_val == 1.0
    assert i._right_val == None

    assert i(5.25) == 10.5
    assert i._x2 == 7
    assert i._at_max == True




if __name__ == '__main__':
    # test_dynamic_1()
    # test_dynamic_3()
    
    # test_dynamic_4()

    # test_dynamic_5()

    # test_dynamic_6()

    # test_dynamic_extensive()


    test_dynamic_3_compare()
