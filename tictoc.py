from time import time

_start_times = {}


def tic(name=None, say_name=False):
    if say_name:
        print(name)
    _start_times[name] = time()
    return _start_times[name]


def toc(name=None):
    if name not in _start_times:
        argument = repr(name) if name is None else ''
        raise RuntimeError(f"Error using toc: You must call tic({argument}) before calling toc({argument})")
    dt = time() - _start_times[name]
    if name is None:
        print(f"Elapsed time: {dt:.6f} seconds")
    else:
        print(f"{name} took {dt:.6f} seconds")
