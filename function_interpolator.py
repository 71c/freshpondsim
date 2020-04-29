from sortedcontainers import SortedDict
import numpy as np
import math
from time import time
from tictoc import tic, toc
from collections import deque


class UnboundedInterpolator:
    """Class that can linearly interpolate through a function that is costly
    to compute, on the go, with no need to specify bounds or pre-compute
    It is costly to do the binary search though so I would recommend using
    BoundedInterpolator instead."""
    def __init__(self, func, resolution, debug=False):
        self._func = func
        self._resolution = resolution
        self._data = SortedDict()
        self._keys = self._data.keys()
        self._debug = debug
        # vectorized function so it can take ndarrays
        self._vf = np.vectorize(self._eval)

    def min_val(self):
        return self._keys[0]

    def max_val(self):
        return self._keys[-1]

    def __call__(self, x):
        if type(x) is np.ndarray:
            return self._vf(x)
        return self._eval(x)

    def _eval(self, x):
        if x in self._data:
            return self._data[x]

        # if there are <= 1 data points or if x is less than or greater than all
        # existing keys, always compute the value
        if len(self._data) <= 1 or x < self.min_val() or x > self.max_val():
            if self._debug:
                print("Computing value of function because not enough data or"
                      " bigger or smaller than all other keys")
            self._data[x] = self._func(x)
            return self._data[x]

        # index of smallest key greater than x
        right_index = self._data.bisect_left(x)
        # index of largest key less than x
        left_index = right_index - 1

        ldiff = x - self._keys[left_index]
        rdiff = self._keys[right_index] - x

        if max(ldiff, rdiff) > self._resolution:
            # if the biggest distance to a neighbor is to big, compute the value
            if self._debug:
                print("Computing value of function because x value not close"
                      " enough to other keys")
            self._data[x] = self._func(x)
            return self._data[x]
        else:
            # otherwise, can interpolate
            if self._debug:
                print("Interpolating")
            lval = self._data[self._keys[left_index]]
            rval = self._data[self._keys[right_index]]
            return (lval * rdiff + rval * ldiff) / (ldiff + rdiff)


class BoundedInterpolator:
    def __init__(self, func, x_min, x_max, resolution):
        self._x_min = x_min
        self._x_max = x_max

        # number of gaps between samples
        n_gaps = math.ceil((x_max - x_min) / resolution)

        n_samples = n_gaps + 1

        # x samples to evaluate function at
        x = np.linspace(x_min, x_max, num=n_samples, endpoint=True)

        # y samples to interpolate between
        # can also do this: self._data = [func(u) for u in x]
        # it doesn't really matter which one
        self._data = np.vectorize(func)(x)

        # space between two x samples
        self._interval = (x_max - x_min) / n_gaps

        # vectorized function so it can take ndarrays
        self._vf = np.vectorize(self._eval)

        print("Before:", resolution)
        print("After:", self._interval)

    def __call__(self, x):
        if type(x) is np.ndarray:
            return self._vf(x)
        return self._eval(x)

    def _eval(self, x):
        if x < self._x_min:
            raise ValueError("A value is below the interpolation range.")
        if x > self._x_max:
            raise ValueError("A value is above the interpolation range.")

        pos = (x - self._x_min) / self._interval
        k = int(pos)
        if k == pos:
            return self._data[k]
        else:
            lval = self._data[k]
            rval = self._data[k + 1]
            ldiff = pos - k
            rdiff = 1 - ldiff
            return lval * rdiff + rval * ldiff


class DynamicBoundedInterpolator:
    def __init__(self, func, x_min, x_max, resolution, expand_factor=2.0, debug=False):
        self._func = func
        self._dx = resolution
        self._expand_factor = expand_factor
        self._x1 = x_min
        self._x2 = x_min
        self._data = deque([self._func(self._x2)])
        self._debug = debug
        self._expand_right(x_max)

        # vectorized function so it can take ndarrays
        self._vf = np.vectorize(self._eval)

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
        if type(x) is np.ndarray:
            return self._vf(x)
        return self._eval(x)

    def _eval(self, x):
        if x <= self._x1:
            new_x1 = self._x1 + self._expand_factor * (x - self._x1)
            self._expand_left(new_x1)
        elif x >= self._x2:
            new_x2 = self._x2 + self._expand_factor * (x - self._x2)
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

if __name__ == '__main__':
    func = lambda x: 2 * x
    intpu = UnboundedInterpolator(func, 1, True)
    print(intpu(3))
    print(intpu(3.1))
    print(intpu(3.2))
    print(intpu(3.5))
    print(intpu(3.4))
    print(intpu(10))
    print(intpu(8))
    print(intpu(9))
    print(intpu(12))
    print(intpu(11))
    print(intpu(3.33))

    intpb = BoundedInterpolator(func, -20, 20, 1.0)
    print(7, intpb(7))
    print(3.3, intpb(3.3))
    print(3.33, intpb(3.33))

    intpd = DynamicBoundedInterpolator(func, -20, -20, 1.0)
    print(intpd(7))
    print(intpd(7.5))
    print(intpd(21))

    # BoundedInterpolator gives slight round-off errors which is a shame
