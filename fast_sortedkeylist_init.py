"""I wrote this code hoping to improve speed but it turns out it does not really
improve the speed so I'm not using it."""
from sortedcontainers import SortedKeyList
from functools import reduce
from operator import iadd


def identity(value):
    "Identity function."
    return value


@classmethod
def new_from_presorted(cls, iterable, key=identity):
    ret = cls(iterable=None, key=key)

    values = iterable if type(iterable) is list else list(iterable)
    _load = ret._load

    ret._lists = [values[pos:(pos + _load)]
                    for pos in range(0, len(values), _load)]
    ret._keys = [list(map(ret._key, _list)) for _list in ret._lists]
    ret._maxes = [sublist[-1] for sublist in ret._keys]

    ret._len = len(values)
    return ret

SortedKeyList.new_from_presorted = new_from_presorted


def update_presorted(self, iterable):
    _lists = self._lists
    _keys = self._keys
    _maxes = self._maxes
    values = iterable if type(iterable) is list else list(iterable)

    if _maxes:
        if len(values) * 4 >= self._len:
            _lists.append(values)
            values = reduce(iadd, _lists, [])
            values.sort(key=self._key)
            self._clear()
        else:
            _add = self.add
            for val in values:
                _add(val)
            return

    _load = self._load
    _lists.extend(values[pos:(pos + _load)]
                    for pos in range(0, len(values), _load))
    _keys.extend(list(map(self._key, _list)) for _list in _lists)
    _maxes.extend(sublist[-1] for sublist in _keys)
    self._len = len(values)
    del self._index[:]

SortedKeyList.update_presorted = update_presorted





if __name__ == "__main__":
    import random
    from operator import neg, attrgetter
    import timeit
    from freshpondsim import FreshPondPedestrian
    import cProfile

    
    n = 100000

    # key = neg
    # make_one = lambda: random.random()

    # key = lambda x: x.start_time
    key = attrgetter('start_time')
    make_one = lambda: FreshPondPedestrian(
        distance_around=2.5,
        start_pos=0,
        travel_distance=4,
        start_time = random.random() * 5,
        time_delta=4)

    a = [make_one() for _ in range(n)]
    a.sort(key=key)


    # kl = SortedKeyList.new_from_presorted(a, key=key)
    # kl._check()

    
    t_my = timeit.timeit(lambda: SortedKeyList.new_from_presorted(a, key=key), number=100)
    print("Using my function", t_my)

    # t_init = timeit.timeit(lambda: SortedKeyList(a, key=key), number=100)
    # print("Using init", t_init)

    # speedup = t_init / t_my
    # print(f"Performance improvement: {speedup:.3g}x faster")

    

    # pr = cProfile.Profile()
    # pr.enable()
    # for _ in range(100):
    #     # u = SortedKeyList(a, key=key)
    #     u = SortedKeyList.new_from_presorted(a, key=key)
    # pr.disable()
    # pr.print_stats(sort='cumulative')
