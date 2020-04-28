from function_interpolator import BoundedInterpolator, DynamicBoundedInterpolator
import numpy as np
from tictoc import *

def test_dynamic():
    f = lambda x: x + x

    a, b = 0, 10000
    
    i1 = DynamicBoundedInterpolator(f, a, a, 1, expand_factor=2.0, debug=False)

    tic(f'{a} to {b}, always expanding')
    for x in np.arange(a, b, 0.1):
        i1(x)
    tocl()

    i2 = DynamicBoundedInterpolator(f, a, a, 1, expand_factor=2.0, debug=False)
    tic(f'{a} to {b}, already expanded')
    i2(500)
    for x in np.arange(a, b, 0.1):
        i2(x)
    tocl()

    i3 = DynamicBoundedInterpolator(f, b, b, 1, expand_factor=2.0, debug=False)
    tic(f'{b} to {a}, always expanding')
    for x in np.arange(b, a, -0.1):
        i3(x)
    tocl()

    i4 = DynamicBoundedInterpolator(f, b, b, 1, expand_factor=2.0, debug=False)
    tic(f'{b} to {a}, already expanded')
    i4(500)
    for x in np.arange(b, a, -0.1):
        i4(x)
    tocl()


if __name__ == '__main__':
    test_dynamic()
