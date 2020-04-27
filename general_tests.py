from sortedcontainers import SortedList
from tictoc import tic, toc
from random import random


def test_insertion_times():
    values = [random() for _ in range(700)]

    tic('on init')
    sl1 = SortedList(values)
    toc('on init')

    tic('one by one')
    sl2 = SortedList()
    for x in values:
        sl2.add(x)
    toc('one by one')


if __name__ == '__main__':
    test_insertion_times()
