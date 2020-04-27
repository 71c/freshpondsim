from sortedcontainers import SortedDict
import numpy as np


class FunctionInterpolator:
    def __init__(self, func, resolution, debug=False):
        self._func = func
        self._resolution = resolution
        self._data = SortedDict()
        self._keys = self._data.keys()
        self._debug = debug
        # vectorized function so it can take ndarrays
        self._vf = np.vectorize(self._eval)

    def __call__(self, x):
        if type(x) is np.ndarray:
            return self._vf(x)
        return self._eval(x)

    def _eval(self, x):
        if x in self._data:
            return self._data[x]

        # if there are <= 1 data points or if x is less than or greater than all
        # existing keys, always compute the value
        if len(self._data) <= 1 or x < self._keys[0] or x > self._keys[-1]:
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


if __name__ == '__main__':
    func = lambda x: 2 * x
    intp = FunctionInterpolator(func, 1, True)
    print(intp(3))
    print(intp(3.1))
    print(intp(3.2))
    print(intp(3.5))
    print(intp(3.4))
    print(intp(10))
    print(intp(8))
    print(intp(9))
    print(intp(12))
    print(intp(11))

