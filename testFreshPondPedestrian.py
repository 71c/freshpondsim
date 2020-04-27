from freshpondsim import FreshPondPedestrian
from math import isclose
import math
from random import random


def assert_isclose(a, b):
    if not (a == b if a is None or b is None else isclose(a, b)):
        raise AssertionError(f"{a} is not close to {b}")


def assert_intersection_time_expected(distance, args1, args2, expected_time=None):
    p1 = FreshPondPedestrian(distance, *args1)
    p2 = FreshPondPedestrian(distance, *args2)
    assert_isclose(p1.first_intersection_time(p2), p2.first_intersection_time(p1))
    if expected_time is not None:
        assert_isclose(p2.first_intersection_time(p1), expected_time)

    t = p2.first_intersection_time(p1)
    assert_isclose(p2.get_position(t), p1.get_position(t))

    assert p1.intersects(p2) == p2.intersects(p1)
    assert p2.intersects(p1) == (t is not None)


def test_intersection_time():
    assert_intersection_time_expected(6.5, (2.7, 24.18, -1.5, 3.9), (1.2, 10.26, -0.8, 1.9))
    assert_intersection_time_expected(1.9, (1.8, 24.18, -1.5, 3.9), (1, 10.26, -0.8, 1.9))
    
    assert_intersection_time_expected(2.7, (1.51, 1.7, 2.2, -4.9), (2.7, 4, -2.1, 0.5), None)

    assert_intersection_time_expected(1.8, (0.85, 24.18, 1.1, -4.9), (1.37, 4, -2.1, -0.7), 1.5095238095238095)

    assert_intersection_time_expected(4.4, (0.5, 24.2, 1.1, -4.9), (1.4, 4, -2.1, 1.2), None)

    assert_intersection_time_expected(4.4, (0.5, 24.2, 1.1, 1), (1.4, 4, -2.1, 1), None)

    # test zero velocity
    assert_intersection_time_expected(4.4, (0.5, 0, 1.1, None, 100), (1.4, 4, -2.1, 1), 1.4)
    assert_intersection_time_expected(1, (0.5, 24.2, 1.1, None, 100), (0.3, 4, -2.1, 1))

    d = math.pi
    for _ in range(10000):
        a = (d * random(), d * random() * 2, 5 * random(), random() * 4 - 2)
        b = (d * random(), d * random() * 2, 5 * random(), random() * 4 - 2)
        assert_intersection_time_expected(d, a, b)


def test_init_edge_cases():
    print("Trying to make zero velocity pedestrian.")
    try:
        p = FreshPondPedestrian(distance_around=DISTANCE, start_pos=0, travel_distance=0, start_time=0, velocity=0)
    except ValueError as e:
        print("Error message:", e)


if __name__ == '__main__':
    test_intersection_time()
    test_init_edge_cases()
