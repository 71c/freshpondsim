from scipy.special import expit
from random import random
from math import exp, log
import numpy as np
from scipy.stats import lognorm
from freshpondsim import FreshPondSim, FreshPondPedestrian
import cProfile
from time import time
import matplotlib.pyplot as plt

def logistic(x, x0, k):
    """logistic function with range (0, 1)
    x: input value
    x0: shift parameter
    k: rate parameter / maximum derivative
    """
    return expit(4 * k * (x - x0))


DISTANCE = 2.46

# counter-clockwise
ENTRANCES_AND_WEIGHTS = [
    (0, 0.7), # vassal ln
    (0.15, 0.5), # concord ave
    (0.2, 0.4), # lusitania field
    (0.39, 0.4), # lusitania field
    (0.42, 0.2), # lusitania field
    (2.3, 0.5) # cambridge water department
]


# meters per second
WALK_SPEED_MEAN = 1.26
WALK_SPEED_STD = 0.09636914420286412

RUN_SPEED_MU = 6.47987998337011
RUN_SPEED_SIGMA = 0.2730453477555863
RUN_SPEED_SCALE = exp(RUN_SPEED_MU)

RUN_PROB = 0.2 # proportion of people who run
# proportion of pepole who are stationary for some proportion of the time
# meaning, who do a mix of walking and sitting/staying
# this is roughly approximated by "walking" very slowly
IDLE_PROB = 0.2
MIN_IDLE_PROPORTION = 0.2
MAX_IDLE_PROPORTION = 0.8
# proportion of non-runners who are stationary for some proportion of the time
IDLE_PROB_GIVEN_NOT_RUN = IDLE_PROB / (1 - RUN_PROB)


def meters_per_second_to_miles_per_minute(x):
    return x / 26.8224


def rand_walk_speed(n=None):
    """returns random walking velocity in miles per minute, can be negative"""
    
    # random walking speed in meters per second
    speed = np.random.normal(loc=WALK_SPEED_MEAN, scale=WALK_SPEED_STD, size=n)
    
    idle_proportion = np.random.uniform(MIN_IDLE_PROPORTION, MAX_IDLE_PROPORTION, n)
    if n is None:
        speed *= 1 if random() < 0.5 else -1
        if random() > IDLE_PROB_GIVEN_NOT_RUN:
            idle_proportion = 0
    else:
        speed[np.random.random(n) < 0.5] *= -1
        idle_proportion[np.random.random(n) > IDLE_PROB_GIVEN_NOT_RUN] = 0
    speed = speed * (1 - idle_proportion)
    return meters_per_second_to_miles_per_minute(speed)


def rand_run_speed(n=None):
    """returns random walking velocity in miles per minute, can be negative"""
    mile_time_secs = lognorm.rvs(RUN_SPEED_SIGMA, loc=0, scale=RUN_SPEED_SCALE, size=n)
    return np.random.choice([-1, 1], n) * 1 / (mile_time_secs / 60)


def rand_velocity(n=None):
    """returns random velocity in miles per minute, can be negative"""
    if n == None:
        if random() < RUN_PROB:
            return rand_run_speed()
        return rand_walk_speed()

    n_runs = round(n * RUN_PROB)
    runs = rand_run_speed(n_runs)
    walks = rand_walk_speed(n - n_runs)
    together = np.concatenate((runs, walks))
    np.random.shuffle(together)
    return together



def get_double_logistic_day_rate_func(min_λ, max_λ, rise_time, rise_rate, fall_time, fall_rate):
    def f(t):
        tmp = logistic(t % 1440, rise_time, rise_rate) * logistic(t % 1440, fall_time, -fall_rate)
        return (max_λ - min_λ) * tmp + min_λ
    return f


def rand_distance_prop():
    if random() < 0.5:
        return 1.0
    if random() < 0.8:
        if random() < 0.4:
            return (2*random()-1) * 0.3 + 0.9
        return random()
    if random() < 0.7:
        return random() + 1
    return random() + 2



def main():
    min_λ, max_λ = 0.01, 1.1185034511
    t1, t2 = 8*60, 19*60
    rate1, rate2 = 1/120, 1/120
    λ = get_double_logistic_day_rate_func(min_λ, max_λ, t1, rate1, t2, rate2)

    entrances, entrance_weights = zip(*ENTRANCES_AND_WEIGHTS)

    t = time()
    sim = FreshPondSim(DISTANCE, 0, 14400, entrances, entrance_weights, λ, rand_velocity, rand_distance_prop)

    print(time() - t)



    # cProfile.runctx('FreshPondSim(DISTANCE, 0, 14400, entrances, entrance_weights, λ, rand_velocity, rand_distance_prop)', globals(), locals(), sort='cumulative')

    # print(max(sim.pedestrians, key=lambda p: p.end_time - p.start_time))

    # t = time()
    # # for _ in range(10000):
    #     # (rand_velocity())
    # u = rand_velocity(10000)
    # print(time() - t)

    # plt.hist(abs(1/u), bins='auto')
    # plt.show()

    durations = [log(p.end_time - p.start_time) for p in sim.pedestrians]
    plt.hist(durations, bins='auto')
    plt.show()

    # distances = [p.travel_distance for p in sim.pedestrians]
    # plt.hist(distances, bins='auto')
    # plt.show()

    # print(len(sim.pedestrians))

    # props = [rand_distance_prop() for _ in range(50000)]
    # plt.hist(props, bins='auto')
    # plt.show()





if __name__ == '__main__':
    main()
