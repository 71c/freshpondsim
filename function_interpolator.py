from sortedcontainers import SortedDict

class FunctionInterpolator:
    def __init__(self, func, resolution):
        self.func = func
        self.resolution = resolution
        self.data = SortedDict()

    def __call__(self, val):
        if val in self.data:
            return self.data[val]
        right_index = self.data.bisect_left(val)
        return self.func(val)

if __name__ == '__main__':
    # func = lambda x: x * x
    # intp = FunctionInterpolator(func, 1)
    # print(intp(3))

    # d = SortedDict()
    # d[7] = 1
    # d[9] = 4
    # d[12] = 6
    # print(d.bisect_left(3))
    # print(d.bisect_left(7))
    # print(d.bisect_left(8))

