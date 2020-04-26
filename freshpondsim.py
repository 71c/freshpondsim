import random
import math
from numbers import Real
from pynverse import inversefunc
import scipy.integrate as integrate


def is_real(x):
    return isinstance(x, Real) and math.isfinite(x)


class FreshPondPedestrian:
    def __init__(self, start_pos, travel_distance, start_time, velocity, distance_around):
        assert is_real(start_pos) and 0 <= start_pos <= distance_around
        assert is_real(travel_distance) and travel_distance >= 0
        assert is_real(start_time)
        assert is_real(velocity)
        assert is_real(distance_around) and distance_around > 0
        self.start_pos = start_pos
        self.start_time = start_time
        self.end_time = math.inf if velocity == 0 else start_time + travel_distance / abs(velocity)
        self.velocity = velocity
        self.distance_around = distance_around
        self.travel_distance = travel_distance

    def _is_in_range(self, t):
        return self.start_time <= t <= self.end_time or math.isclose(t, self.start_time) or math.isclose(t, self.end_time)

    def get_position(self, t):
        if t is None:
            return None
        if not self._is_in_range(t):
            return None
        return (self.start_pos + (t - self.start_time) * self.velocity) % self.distance_around

    def intersection_time(self, other):
        """Returns first time when my pos = other pos or None if no such time"""
        assert self.distance_around == other.distance_around
        if self.velocity == other.velocity:
            return None
        min_time = max(self.start_time, other.start_time)
        max_time = min(self.end_time, other.end_time)
        if min_time > max_time:
            return None

        x01, t01, v1 = self.start_pos, self.start_time, self.velocity
        x02, t02, v2 = other.start_pos, other.start_time, other.velocity

        tmp = t01*v1 - t02*v2 + x02 - x01
        k1 = (min_time * (v1 - v2) - tmp) / self.distance_around
        k2 = (max_time * (v1 - v2) - tmp) / self.distance_around

        # try to make k as close to possible to k1
        # min_time <= t <= max_time --> inequality with k
        if v1 > v2:
            # k1 <= k <= k2
            # and try to make k as close to possible to k1
            k = math.ceil(k1)
        else:
            # k1 >= k >= k2 (the signs flip when solving)
            k = math.floor(k1)

        t = (k * self.distance_around + tmp) / (v1 - v2)
        if t > max_time:
            return None
        return t

    def __repr__(self):
        return(f'FreshPondPedestrian(start_time={self.start_time:.2f}, end_time={self.end_time:.2f}, start_pos={self.start_pos}, mile_time={1/self.velocity:.2f})')


def newtons_method(x0, func, d_func, tol, max_iter):
    x = x0
    for _ in range(max_iter):
        y = func(x)
        x -= y / d_func(x)
        if abs(y) < tol:
            break
    return x


def rand_next_time(t0, λfunc):
    a = -math.log(random.random())

    def L(t):
        y, abserr = integrate.quad(λfunc, t0, t)
        return y - a

    tol = 1e-4

    x = newtons_method(t0, L, λfunc, tol, 8)

    if abs(L(x)) > tol:
        inv_L = inversefunc(L, y_values=[0], accuracy=5)
        x = inv_L[0]

    return x


def random_times(t0, t_max, λfunc):
    t = rand_next_time(t0, λfunc)
    while t < t_max:
        yield t
        t = rand_next_time(t, λfunc)


def circular_difference(a, b, max_val):
    c = a - b
    c = (c + max_val * 0.5) % max_val - max_val * 0.5
    return c


class FreshPondSim:
    def __init__(self, distance, start_time, stop_time, entrances, entrance_weights, entrance_rate_func, rand_velocity_func, rand_distance_prop_func):
        assert is_real(distance) and distance > 0
        assert is_real(start_time)
        assert is_real(stop_time)
        assert start_time < stop_time
        self.start_time = start_time
        self.stop_time = stop_time
        self.distance_around = distance
        assert len(entrances) == len(entrance_weights)
        self.entrances = entrances
        self.entrance_weights = entrance_weights
        self.entrance_rate_func = entrance_rate_func
        self.rand_velocity = rand_velocity_func
        self.rand_distance_prop = rand_distance_prop_func

        self.pedestrians = []
        self.initialize_pedestrians()

    def initialize_pedestrians(self):
        # start_times = list(random_times(self.start_time, self.stop_time, self.entrance_rate_func))
        # entrances = random.choices(population=self.entrances, weights=self.entrance_weights, k=len(start_times))
        # for start_time, entrance in zip(start_times, entrances):
        #     proposed_distance = self.rand_distance_prop() * self.distance_around
        #     closest_exit = min(self.entrances, key=lambda x: abs(circular_difference(proposed_distance % self.distance_around, x, self.distance_around)))
        #     diff = circular_difference(proposed_distance % self.distance_around, closest_exit, self.distance_around)
        #     distance = proposed_distance - diff
        #     if math.isclose(distance, 0, abs_tol=1e-10):
        #         distance = self.distance_around
        #     velocity = self.rand_velocity()
        #     self.pedestrians.append(FreshPondPedestrian(entrance, distance, start_time, velocity, self.distance_around))

        start_times = list(random_times(self.start_time, self.stop_time, self.entrance_rate_func))
        n_pedestrians = len(start_times)
        entrances = random.choices(population=self.entrances, weights=self.entrance_weights, k=n_pedestrians)
        velocities = self.rand_velocity(n_pedestrians)

        for start_time, entrance, velocity in zip(start_times, entrances, velocities):
            proposed_distance = self.rand_distance_prop() * self.distance_around
            closest_exit = min(self.entrances, key=lambda x: abs(circular_difference(proposed_distance % self.distance_around, x, self.distance_around)))
            diff = circular_difference(proposed_distance % self.distance_around, closest_exit, self.distance_around)
            distance = proposed_distance - diff
            if math.isclose(distance, 0, abs_tol=1e-10):
                distance = self.distance_around
            self.pedestrians.append(FreshPondPedestrian(entrance, distance, start_time, velocity, self.distance_around))











