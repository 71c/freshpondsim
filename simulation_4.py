import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
from scipy.stats import lognorm
import scipy.integrate as integrate
from tictoc import *


def get_sim_with_mile_time_dist(distance, dt, entrance_rate, mile_time_dist, n_times_around,
                   same_direction_prop):
    rate_func = lambda t: entrance_rate
    rate_integral_func = lambda t: entrance_rate * t

    scale = 1/mile_time_mean * np.exp(sigma**2 / 2) # exp(mu)

    def get_vdist(n):
        ret = np.array([mile_time_dist.rvs(size=n), np.ones(n) * n_times_around * distance]).T
        ret[np.random.random(n) > same_direction_prop, 0] *= -1
        return ret

    return FreshPondSim(distance,
                        0,
                        dt, [0], [1],
                        get_vdist,
                        rate_func,
                        entrance_rate_integral=rate_integral_func,
                        interpolate_rate=False,
                        interpolate_rate_integral=False,
                        snap_exit=False)


distance = 2
dt = 400
entrance_rate = 4
mile_time_mean = 20
sigma = 0.8
n_times_around = 2
same_direction_prop = 0.3

scale = 1/mile_time_mean * np.exp(sigma**2 / 2) # exp(mu)
mile_time_dist = lognorm(sigma, loc=0, scale=scale)

sim = get_sim_with_mile_time_dist(distance, dt, entrance_rate, mile_time_dist, n_times_around, same_direction_prop)

mean_duration = mile_time_mean * distance * n_times_around

n_times_around_observer = 0.2
observer_mile_time = 10
observer = FreshPondPedestrian(
    distance_around=distance,
    start_pos=0,
    travel_distance=n_times_around_observer * distance,
    start_time=4 * mean_duration,
    velocity=1/observer_mile_time)

tic()

mean_n_same, mean_n_diff = 0.0, 0.0
n_trials = 50
for i in range(n_trials):
    n_same, n_diff = sim.intersection_directions(observer)
    mean_n_same += n_same
    mean_n_diff += n_diff
    sim.refresh_pedestrians()
mean_n_same /= n_trials
mean_n_diff /= n_trials

toc()


print(f"mean_duration:", mean_duration)
expected_n_people = entrance_rate * mean_duration
print(expected_n_people, sim.n_people(4 * mean_duration))


# scale = 1/mile_time_mean * np.exp(sigma**2 / 2) # exp(mu)
# dist = lognorm(sigma, loc=0, scale=scale)
expected_n_same = same_direction_prop * expected_n_people * n_times_around_observer * mile_time_dist.expect(lambda v: np.abs(observer_mile_time * v - 1))

wrong_expected_n_same = same_direction_prop * n_times_around_observer * expected_n_people * mile_time_dist.expect(lambda v: np.abs(1/v - observer_mile_time)) / mile_time_mean

print('expected n same:', expected_n_same)
print('wrong expected n same:', wrong_expected_n_same)
print('mean n same:', mean_n_same)
