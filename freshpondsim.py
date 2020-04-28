import random
import math
from numbers import Real
from pynverse import inversefunc
import scipy.integrate as integrate
from sortedcontainers import SortedList, SortedDict
from tictoc import tic, toc
from function_interpolator import BoundedInterpolator, DynamicBoundedInterpolator
from scipy.interpolate import interp1d
import numpy as np


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
    def __init__(self, distance_around, start_pos, travel_distance, start_time, velocity=None, time_delta=None):
        """
        distance_around: distance around the path in miles
        start_pos: starting position in miles
        travel_distance: distance traveled.
            If velocity is given, travel_distance must be positive because the
            direction is specified in the velocity.
            If time_delta is given, the sign of travel_distance indicates the
            direction of travel.
        start_time: the starting time
        velocity: velocity in miles per minute
        time_delta: time taken in minutes
        """
        assert_positive_real(distance_around, 'distance_around')
        if not (is_real(start_pos) and 0 <= start_pos <= distance_around):
            raise ValueError(f"start_pos {start_pos} is not a number in range [0, distance_around]")
        assert_real(start_time, 'start_time')

        self.distance_around = distance_around
        self.start_pos = start_pos
        self.travel_distance = abs(travel_distance) # not necessary to have
        self.start_time = start_time

        if velocity is not None and time_delta is not None:
            raise ValueError("Don't specify both velocity and time_delta")
        if velocity is not None:
            assert_nonnegative_real(travel_distance, 'travel_distance')
            assert_real(velocity, 'velocity')
            if travel_distance == 0:
                raise ValueError("Travel distance cannot be zero if specifying velocity. Specify time_delta instead.")
            # travel_distance != 0
            if velocity == 0:
                raise ValueError("Velocity cannot be zero")
            self.end_time = start_time + travel_distance / abs(velocity)
            self.velocity = velocity
        elif time_delta is not None:
            assert_real(travel_distance, 'travel_distance')
            assert_positive_real(time_delta, 'time_delta')
            self.end_time = start_time + time_delta
            self.velocity = travel_distance / time_delta
        else:
            raise ValueError("Specify either velocity or time_delta")

    def is_in_range(self, t):  
        """Returns whether t is in the time range of this pedestrian
        The time range is a half open interval"""
        return self.start_time <= t < self.end_time

    def get_position(self, t):
        if t is None:
            return None
        if not (self.is_in_range(t) or math.isclose(t, self.start_time)):
            return None
        return (self.start_pos + (t - self.start_time) * self.velocity) % self.distance_around

    def first_intersection_time(self, other):
        """Returns first time when my pos = other pos or None if no such time"""
        assert self.distance_around == other.distance_around
        if self.end_time <= other.start_time or self.start_time >= other.end_time:
            return None
        if self.velocity == other.velocity:
            return None
        # an intersection time must be at least min_time
        min_time = max(self.start_time, other.start_time)
        # an intersection time must be less than max_time
        max_time = min(self.end_time, other.end_time)

        x01, t01, v1 = self.start_pos, self.start_time, self.velocity
        x02, t02, v2 = other.start_pos, other.start_time, other.velocity

        tmp = t01*v1 - t02*v2 + x02 - x01
        k1 = (min_time * (v1 - v2) - tmp) / self.distance_around
        k2 = (max_time * (v1 - v2) - tmp) / self.distance_around

        # try to make k as close to possible to k1
        # min_time <= t < max_time --> inequality with k
        if v1 > v2:
            # k1 <= k < k2
            # and try to make k as close to possible to k1
            k = math.ceil(k1)
        else:
            # k1 >= k > k2 (the signs flip when solving)
            k = math.floor(k1)

        t = (k * self.distance_around + tmp) / (v1 - v2)
        if t >= max_time:
            return None
        return t

    def intersects(self, other):
        """Returns whether self ever crosses other's path"""

        # can also do this but it is slower
        # return self.intersection_time(other) is not None

        assert self.distance_around == other.distance_around
        if self.end_time <= other.start_time or self.start_time >= other.end_time:
            return False
        if self.velocity == other.velocity:
            return False
        # an intersection time must be at least min_time
        min_time = max(self.start_time, other.start_time)
        # an intersection time must be less than max_time
        max_time = min(self.end_time, other.end_time)

        x01, t01, v1 = self.start_pos, self.start_time, self.velocity
        x02, t02, v2 = other.start_pos, other.start_time, other.velocity

        tmp = t01*v1 - t02*v2 + x02 - x01
        k1 = (min_time * (v1 - v2) - tmp) / self.distance_around
        k2 = (max_time * (v1 - v2) - tmp) / self.distance_around

        floor_k1 = math.floor(k1)
        floor_k2 = math.floor(k2)
        # whether there is an integer in range [k1, k2)
        return floor_k1 != floor_k2 or floor_k1 == k1

    def n_intersections(self, other):
        # TODO
        pass

    def get_mile_time(self):
        return math.inf if self.velocity == 0 else 1 / self.velocity

    def __repr__(self):
        return(f'FreshPondPedestrian(start_time={self.start_time:.2f}, end_time={self.end_time:.2f}, start_pos={self.start_pos}, mile_time={self.get_mile_time():.2f})')


def newtons_method(x0, func, d_func, tol, max_iter):
    x = x0
    for _ in range(max_iter):
        y = func(x)
        x -= y / d_func(x)
        if abs(y) < tol:
            break
    return x


def rand_next_time(t0, λfunc, λfunc_integral=None):
    a = -math.log(random.random())

    if λfunc_integral is None:
        def L(t):
            y, abserr = integrate.quad(λfunc, t0, t)
            return y - a
    else:
        t0_integral = λfunc_integral(t0)
        def L(t):
            y = λfunc_integral(t) - t0_integral
            return y - a

    tol = 1e-4

    x = newtons_method(t0, L, λfunc, tol, 20)

    if abs(L(x)) > tol:
        inv_L = inversefunc(L, y_values=[0], accuracy=5)
        x = inv_L[0]

    return x


def random_times(t0, t_max, λfunc, λfunc_integral=None):
    t = rand_next_time(t0, λfunc, λfunc_integral)
    while t < t_max:
        yield t
        t = rand_next_time(t, λfunc, λfunc_integral)


def circular_diff(a, b, max_val):
    c = a - b
    c = (c + max_val * 0.5) % max_val - max_val * 0.5
    return c


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


class FreshPondSim:
    def __init__(self, distance, start_time, end_time, entrances, entrance_weights, entrance_rate_func, rand_rand_velocities_and_distances_func, interpolate=False, interpolate_res=None):
        assert_positive_real(distance, 'distance')
        assert_real(start_time, 'start_time')
        assert_real(end_time, 'end_time')
        if not (start_time < end_time):
            raise ValueError(f"start_time should be less than end_time")
        assert len(entrances) == len(entrance_weights)
        self.start_time = start_time
        self.end_time = end_time
        self.dist_around = distance
        self.entrances = entrances
        self.entrance_weights = entrance_weights
        self.entrance_rate_func = entrance_rate_func
        self.rand_velocities_and_distances = rand_rand_velocities_and_distances_func

        if interpolate:
            if interpolate_res is None:
                raise ValueError("Specify interpolate_res for interpolation")
            def integral_func(t):
                y, abserr = integrate.quad(entrance_rate_func, start_time, t)
                return y

            # Time periods can go above the interpolation range so give the
            # interpolation function some more space at the end
            # interpolation_min = start_time - (end_time - start_time) * 0.1
            # interpolation_max = end_time + (end_time - start_time) * 0.5

            # self.entrance_rate_func_integral = BoundedInterpolator(integral_func, interpolation_min, interpolation_max, interpolate_res)
            self.entrance_rate_func_integral = DynamicBoundedInterpolator(integral_func, start_time, end_time, interpolate_res)

            # can also interpolate this one! it does make it somewhat fater!
            # self.entrance_rate_func = BoundedInterpolator(entrance_rate_func, interpolation_min, interpolation_max, interpolate_res)
            # self.entrance_rate_func = DynamicBoundedInterpolator(entrance_rate_func, start_time, end_time, interpolate_res)

            

        else:
            self.entrance_rate_func_integral = None

        self.pedestrians = SortedList(key=lambda p: p.start_time)
        self._counts = SortedDict()
        self.set_random_pedestrians()

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

    def set_random_pedestrians(self):
        """Refreshes the pedestrians in the simulation to random ones"""
        self.clear_pedestrians()
        
        start_times = list(random_times(self.start_time, self.end_time, self.entrance_rate_func, self.entrance_rate_func_integral))
        n_pedestrians = len(start_times)
        entrances = random.choices(population=self.entrances, weights=self.entrance_weights, k=n_pedestrians)
        velocities, distances = self.rand_velocities_and_distances(n_pedestrians).T
        
        def pedestrians_generator():
            for start_time, entrance, velocity, dist in zip(start_times, entrances, velocities, distances):
                assert dist > 0
                original_exit = entrance + dist * sign(velocity)
                corrected_exit = self._closest_exit(original_exit)
                corrected_dist = abs(corrected_exit - entrance)
                if math.isclose(corrected_dist, 0, abs_tol=1e-10):
                    corrected_dist = self.dist_around
                yield FreshPondPedestrian(self.dist_around, entrance, corrected_dist, start_time, velocity)

        self.add_pedestrians(pedestrians_generator())

    def clear_pedestrians(self):
        """Removes all pedestrains in the simulation"""
        self.pedestrians.clear()
        self._reset_counts()

    def add_pedestrians(self, pedestrians):
        """Adds all the given pedestrians to the simulation"""
        def checked_pedestrians():
            for p in pedestrians:
                self._assert_pedestrian_in_range(p)
                yield p
        self.pedestrians.update(checked_pedestrians())
        self._recompute_counts()

    def _assert_pedestrian_in_range(self, p):
        """Makes sure the pedestrian's start time is in the simulation's
        time interval"""
        if not (self.start_time <= p.start_time < self.end_time):
            raise ValueError("Pedestrian start time is not in range [start_time, end_time)")

    def add_pedestrian(self, p):
        """Adds a new pedestrian to the simulation"""
        self._assert_pedestrian_in_range(p)
        self.pedestrians.add(p)

        # add a new breakpoint at the pedestrian's start time if it not there
        self._counts[p.start_time] = self.n_people(p.start_time)

        # add a new breakpoint at the pedestrian's end time if it not there
        self._counts[p.end_time] = self.n_people(p.end_time)

        # increment all the counts in the pedestrian's interval of time
        # inclusive on the left, exclusive on the right
        # If it were inclusive on the right, then the count would be one more
        # than it should be in the period after end_time and before the next
        # breakpoint after end_time
        for t in self._counts.irange(p.start_time, p.end_time, inclusive=(True, False)):
            self._counts[t] += 1

    def _reset_counts(self):
        """Clears _counts and sets count at start_time to 0"""
        self._counts.clear()
        self._counts[self.start_time] = 0

    def _recompute_counts(self):
        """Store how many people there are whenever someone enters or exits so
        the number of people at a given time can be found quickly later"""
        self._reset_counts()

        if self.num_pedestrians() == 0:
            return

        start_times = [] # pedestrians are already sorted by start time
        end_times = SortedList()
        for pedestrian in self.pedestrians:
            start_times.append(pedestrian.start_time)
            end_times.add(pedestrian.end_time)

        n = len(start_times)
        curr_count = 0 # current number of people
        start_times_index = 0
        end_times_index = 0
        starts_done = False # whether all the start times have been added
        ends_done = False # whether all the end times have been added
        while not (starts_done and ends_done):
            # determine whether a start time or an end time should be added next
            # store this in the variable take_start which is true if a start
            # time should be added next
            if starts_done:
                # already added all the start times; add an end time
                take_start = False
            elif ends_done:
                # already added all the end times; add a start time
                take_start = True
            else:
                # didn't add all the end times nor all the start times
                # add the time that is earliest
                next_start_time = start_times[start_times_index]
                next_end_time = end_times[end_times_index]
                take_start = next_start_time < next_end_time

            if take_start:
                # add next start
                curr_count += 1
                start_time = start_times[start_times_index]
                self._counts[start_time] = curr_count
                start_times_index += 1
                if start_times_index == n:
                    starts_done = True
            else:
                # add next end
                curr_count -= 1
                end_time = end_times[end_times_index]
                self._counts[end_time] = curr_count
                end_times_index += 1
                if end_times_index == n:
                    ends_done = True

    def n_unique_people_saw(self, p):
        """Returns the number of unique people that a pedestrian sees"""
        n = 0
        for q in self.pedestrians:
            if p.intersects(q):
                n += 1
        return n

    def n_people(self, t):
        """Returns the number of people at a given time"""
        if t in self._counts:
            return self._counts[t]
        elif t < self.start_time:
            return 0
        else:
            index = self._counts.bisect_left(t)
            return self._counts.values()[index - 1]

    def num_pedestrians(self):
        """Returns the total number of pedestrians in the simulation"""
        return len(self.pedestrians)

