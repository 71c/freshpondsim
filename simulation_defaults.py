from scipy.special import expit
from random import random
import numpy as np
from scipy.stats import lognorm
from freshpondsim import FreshPondSim


def _logistic(x, x0, k):
    """_logistic function with range (0, 1)
    x: input value
    x0: shift parameter
    k: rate parameter / maximum derivative
    """
    return expit(4 * k * (x - x0))


DISTANCE = 2.46

# counter-clockwise
ENTRANCES_AND_WEIGHTS = [
    (0, 0.7),  # vassal ln
    (0.15, 0.5),  # concord ave
    (0.2, 0.4),  # lusitania field
    (0.39, 0.4),  # lusitania field
    (0.42, 0.2),  # lusitania field
    (2.3, 0.5)  # cambridge water department
]
ENTRANCES, ENTRANCE_WEIGHTS = zip(*ENTRANCES_AND_WEIGHTS)

## Normalize distance or set to a desired value to make it easier to see trends
# ENTRANCES = np.array(ENTRANCES) / DISTANCE
# ENTRANCE_WEIGHTS = np.array(ENTRANCE_WEIGHTS) / DISTANCE
# DISTANCE = 1
# ENTRANCES *= DISTANCE
# ENTRANCE_WEIGHTS *= DISTANCE

## Only one entrance -- to simplify further
# ENTRANCES = ENTRANCES[0:1]
# ENTRANCE_WEIGHTS = ENTRANCE_WEIGHTS[0:1]

# miles per minute
WALK_SPEED_MEAN = 0.04697566213
WALK_SPEED_STD = 0.003592860602

RUN_SPEED_MU = 2.38553542115
RUN_SPEED_SIGMA = 0.2730453477555863
RUN_SPEED_SCALE = np.exp(RUN_SPEED_MU)

RUN_PROB = 0.2  # proportion of people who run
MIN_IDLE_PROPORTION = 0.2
MAX_IDLE_PROPORTION = 0.8

MAX_IDLE_TIME = 60 * 4

DAY_LENGTH = 1440


def get_double_logistic_day_rate_func(min_λ, max_λ, rise_time, rise_rate,
                                      fall_time, fall_rate):
    def f(t):
        tmp = _logistic(t % DAY_LENGTH, rise_time, rise_rate) * _logistic(
            t % DAY_LENGTH, fall_time, -fall_rate)
        return (max_λ - min_λ) * tmp + min_λ

    return f


def _get_default_day_rate_func():
    min_λ, max_λ = 0.01, 1.1185034511
    t1, t2 = 8 * 60, 19 * 60
    rate1, rate2 = 1 / 120, 1 / 120
    return get_double_logistic_day_rate_func(min_λ, max_λ, t1, rate1, t2, rate2)


default_day_rate_func = _get_default_day_rate_func()

constant_day_rate_func = lambda x: 1


def _rand_distance_prop():
    if random() < 0.9:
        n_times = 0
    elif random() < 0.7:
        n_times = 1
    else:
        n_times = 2
    if random() < 0.5:
        return n_times + 1
    if random() < 0.4:
        return np.random.normal(loc=n_times + 1, scale=0.12)
    return n_times + random()


def _soft_minimum_positive(x, x_max, beta):
    tmp1 = np.log(1 + np.exp((1 - x / x_max) / beta))
    tmp2 = np.log(1 + np.exp(1 / beta))
    return x_max * (1 - tmp1 / tmp2)


def rand_walk_velocities_and_distances(n):
    dist_props = np.array([_rand_distance_prop() for _ in range(n)])
    dists = dist_props * DISTANCE

    a, b, c = 0.395, 1.1, 1.0
    idle_probs = a * np.exp(-(dist_props / b)**c)

    # random walking speeds in miles per minute
    speeds = np.random.normal(loc=WALK_SPEED_MEAN, scale=WALK_SPEED_STD, size=n)

    # 'idle' means pepole who are stationary for some proportion of the time,
    # meaning who do a mix of walking and sitting/staying
    # this is roughly approximated by "walking" very slowly

    # get random idle proportions
    idle_proportions = np.random.uniform(MIN_IDLE_PROPORTION,
                                         MAX_IDLE_PROPORTION, n)
    # make some of them 0
    idle_proportions[np.random.random(n) > idle_probs] = 0

    # maximum idle proportions such that the time idle <= MAX_IDLE_TIME
    max_idle_proportions = 1 / (1 + dists / (MAX_IDLE_TIME * speeds))
    # constrain idle_proportions so that the times idle are not too long
    # idle_proportions = np.minimum(idle_proportions, max_idle_proportions)
    idle_proportions = _soft_minimum_positive(idle_proportions,
                                              max_idle_proportions, 0.1)

    speeds = speeds * (1 - idle_proportions)

    # make speeds have random signs
    speeds[np.random.random(n) < 0.5] *= -1

    return np.stack((speeds, dists)).T


def rand_run_speed(n=None):
    """returns random walking velocity in miles per minute, can be negative"""
    mile_time_mins = lognorm.rvs(RUN_SPEED_SIGMA,
                                 loc=0,
                                 scale=RUN_SPEED_SCALE,
                                 size=n)
    return np.random.choice([-1, 1], n) / mile_time_mins


def rand_run_velocities_and_distances(n):
    dist_props = np.array([_rand_distance_prop() for _ in range(n)])
    dists = dist_props * DISTANCE
    speeds = rand_run_speed(n)
    return np.stack((speeds, dists)).T


def default_rand_velocities_and_distances(n):
    n_runs = np.random.binomial(n, RUN_PROB)
    n_walks = n - n_runs

    walks = rand_walk_velocities_and_distances(n_walks)
    runs = rand_run_velocities_and_distances(n_runs)
    together = np.concatenate((walks, runs), axis=0)
    np.random.shuffle(together)

    return together


def constant_velocity_rand_velocities_and_distances(n):
    # exactly 20 min/mi for easy spotting
    velocities = np.random.choice([-1, 1], n) * (1 / 20)
    # dists = np.array([DISTANCE * _rand_distance_prop() for _ in range(n)])
    dists = np.array([1 * DISTANCE for _ in range(n)])
    return np.stack((velocities, dists)).T


def default_sim():
    return FreshPondSim(DISTANCE,
                        0,
                        DAY_LENGTH,
                        ENTRANCES,
                        ENTRANCE_WEIGHTS,
                        default_rand_velocities_and_distances,
                        default_day_rate_func,
                        interpolate_rate=True,
                        interpolate_rate_integral=True,
                        interpolate_res=2.0)


## The next 3 functions are now obsolete

# def sim_constant_rate():
#     return FreshPondSim(DISTANCE,
#                         0,
#                         DAY_LENGTH,
#                         ENTRANCES,
#                         ENTRANCE_WEIGHTS,
#                         default_rand_velocities_and_distances,
#                         rate_func=constant_day_rate_func,
#                         interpolate_rate=True,
#                         interpolate_rate_integral=True,
#                         interpolate_res=2.0)
#
#
# def sim_constant_speed():
#     return FreshPondSim(DISTANCE,
#                         0,
#                         DAY_LENGTH,
#                         ENTRANCES,
#                         ENTRANCE_WEIGHTS,
#                         constant_velocity_rand_velocities_and_distances,
#                         default_day_rate_func,
#                         interpolate_rate=True,
#                         interpolate_rate_integral=True,
#                         interpolate_res=2.0)


def sim_constant_rate_and_speed():
    return FreshPondSim(DISTANCE,
                        0,
                        DAY_LENGTH,
                        ENTRANCES,
                        ENTRANCE_WEIGHTS,
                        constant_velocity_rand_velocities_and_distances,
                        constant_day_rate_func,
                        interpolate_rate=True,
                        interpolate_rate_integral=True,
                        interpolate_res=2.0)
