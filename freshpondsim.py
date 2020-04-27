import random
import math
from numbers import Real
from pynverse import inversefunc
import scipy.integrate as integrate


def is_real(x):
    return isinstance(x, Real) and math.isfinite(x)


def assert_real(val, name):
    if not is_real(val):
        print(f"{name} should be a real number but is {val}")


def assert_nonnegative_real(val, name):
    if not (is_real(val) and val >= 0):
        raise ValueError(f"{name} should be a number in range [0, inf) but is {val}")


def assert_positive_real(val, name):
    if not (is_real(val) and val > 0):
        raise ValueError(f"{name} should be a number in range (0, inf) but is {val}")


class FreshPondPedestrian:
    def __init__(self, start_pos, travel_distance, start_time, velocity, distance_around):
        if not (is_real(start_pos) and 0 <= start_pos <= distance_around):
            raise ValueError(f"start_pos {start_pos} is not a number in range [0, distance_around]")
        assert_nonnegative_real(travel_distance, 'travel_distance')
        assert_real(start_time, 'start_time')
        assert_real(velocity, 'velocity')
        assert_positive_real(distance_around, 'distance_around')
        self.start_pos = start_pos
        self.start_time = start_time
        self.end_time = math.inf if velocity == 0 else start_time + travel_distance / abs(velocity)
        self.velocity = velocity
        self.distance_around = distance_around
        self.travel_distance = travel_distance

    def is_in_range(self, t):  
        return self.start_time <= t <= self.end_time

    def get_position(self, t):
        if t is None:
            return None
        if not (self.is_in_range(t) or math.isclose(t, self.start_time) or math.isclose(t, self.end_time)):
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

    def intersects(self, other):
        return self.intersection_time(other) is not None

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


def circular_diff(a, b, max_val):
    c = a - b
    c = (c + max_val * 0.5) % max_val - max_val * 0.5
    return c


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


class FreshPondSim:
    def __init__(self, distance, start_time, stop_time, entrances, entrance_weights, entrance_rate_func, rand_rand_velocities_and_distances_func):
        assert_positive_real(distance, 'distance')
        assert_real(start_time, 'start_time')
        assert_real(stop_time, 'stop_time')
        if not (start_time < stop_time):
            raise ValueError(f"start_time should be less than stop_time")
        assert len(entrances) == len(entrance_weights)
        self.start_time = start_time
        self.stop_time = stop_time
        self.dist_around = distance
        self.entrances = entrances
        self.entrance_weights = entrance_weights
        self.entrance_rate_func = entrance_rate_func
        self.rand_velocities_and_distances = rand_rand_velocities_and_distances_func
        self._initialize_pedestrians()

    def _distance(self, a, b):
        """signed distance of a relative to b"""
        return circular_diff(a % self.dist_around, b % self.dist_around, self.dist_around)

    def _distance_from(self, b):
        """returns a function that returns the signed sitance from b"""
        return lambda a: self._distance(a, b)

    def _abs_distance_from(self, b):
        """returns a function that returns the distance from b"""
        return lambda a: abs(self._distance(a, b))

    def _closest_exit(self, dist):
        """Returns the closest number to dist that is equivalent mod dist_around
        to an element of entrances"""
        closest_exit = min(self.entrances, key=self._abs_distance_from(dist))
        diff = self._distance(closest_exit, dist)
        corrected_dist = dist + diff
        return corrected_dist

    def _initialize_pedestrians(self):
        self.pedestrians = []
        start_times = list(random_times(self.start_time, self.stop_time, self.entrance_rate_func))
        n_pedestrians = len(start_times)
        entrances = random.choices(population=self.entrances, weights=self.entrance_weights, k=n_pedestrians)
        velocities, distances = self.rand_velocities_and_distances(n_pedestrians).T
        for start_time, entrance, velocity, dist in zip(start_times, entrances, velocities, distances):
            assert dist > 0
            original_exit = entrance + dist * sign(velocity)
            corrected_exit = self._closest_exit(original_exit)
            corrected_dist = abs(corrected_exit - entrance)
            if math.isclose(corrected_dist, 0, abs_tol=1e-10):
                corrected_dist = self.dist_around

            self.pedestrians.append(FreshPondPedestrian(entrance, corrected_dist, start_time, velocity, self.dist_around))

    def n_people_saw(self, p):
        n = 0
        for q in self.pedestrians:
            if p.intersects(q):
                n += 1
        return n

    def n_people(self, t):
        n = 0
        for q in self.pedestrians:
            if q.is_in_range(t):
                n += 1
        return n

