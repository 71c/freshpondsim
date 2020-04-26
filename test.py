from random import random
from math import log, exp
from scipy.special import expit
import scipy.integrate as integrate
from time import time
from pynverse import inversefunc
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import cProfile


def exp_dist_rand(λ):
    return -log(random()) / λ


def rand_times_exponential(λ, n):
    val = 0
    count = 0
    while count < n:
        val += exp_dist_rand(λ)
        yield val
        count += 1


def logistic(x, x0, k):
    """logistic function with range (0, 1)
    x: input value
    x0: shift parameter
    k: rate parameter / maximum derivative
    """
    return expit(4 * k * (x - x0))



MAX_λ = 1.1185034511
MIN_λ = 0.01
# K_PARAM = 0.016
AVG_AROUND_TIME = 45.8
K_PARAM = 1/AVG_AROUND_TIME
DISTANCE = 2.5 # around in miles

# meters per second
WALK_SPEED_MEAN = 1.26
WALK_SPEED_STD = 0.09636914420286412

RUN_SPEED_MU = 6.47987998337011
RUN_SPEED_SIGMA = 0.2730453477555863
RUN_SPEED_SCALE = exp(RUN_SPEED_MU)

RUN_PROB = 0.3

def metersToSecondToMinutesPerMile(x):
    return 26.8224 / x

def rand_walk_time(n=1):
    rand = np.random.randn() if n == 1 else np.random.randn(n)
    speed = WALK_SPEED_MEAN + rand * WALK_SPEED_STD # m/s
    return DISTANCE * metersToSecondToMinutesPerMile(speed)


def rand_run_time(n=1):
    if n == 1:
        n = None
    mile_time_secs = lognorm.rvs(RUN_SPEED_SIGMA, loc=0, scale=RUN_SPEED_SCALE, size=n)
    return mile_time_secs / 60 * DISTANCE


def rand_around_time(n=1):
    if n == 1:
        if random() < RUN_PROB:
            return rand_run_time()
        return rand_walk_time()
    
    n_runs = round(n * RUN_PROB)
    runs = rand_run_time(n_runs)
    walks = rand_walk_time(n - n_runs)
    together = np.concatenate((runs, walks))
    np.random.shuffle(together)
    return together


def my_λ(t):
    tmp = logistic(t % 1440, 8*60, 1/120) * logistic(t % 1440, 19*60, -1/120)
    return (MAX_λ - MIN_λ) * tmp + MIN_λ


def _integrate_this(t):
    return exp(K_PARAM * t) * my_λ(t)


# THIS MODEL IS WRONG!!!
def my_n(t, n0):
    y, abserr = integrate.quad(_integrate_this, 0, t)
    return exp(-K_PARAM * t) * (n0 + y)


def newtons_method(x0, func, d_func, tol, max_iter):
    x = x0
    for _ in range(max_iter):
        y = func(x)
        x -= y / d_func(x)
        if abs(y) < tol:
            break
    return x


def rand_time(t0):
    a = -log(random())

    def L(t):
        # integrate.fixed_quad is faster than integrate.quad
        # it seems the higher n is the more accurate
        y, none = integrate.fixed_quad(my_λ, t0, t, n=100)

        # y, abserr = integrate.quad(my_λ, t0, t)
        return y - a

    tol = 1e-4

    x = newtons_method(t0, L, my_λ, tol, 8)

    if abs(L(x)) > tol:
        inv_L = inversefunc(L, y_values=[0], accuracy=5)
        x = inv_L[0]

    return x


def my_random_times(t0, t_max):
    t = rand_time(t0)
    while t < t_max:
        yield t
        t = rand_time(t)


def get_simulation_events(min_t, max_t):
    starts = []
    ends = []
    for start in my_random_times(min_t, max_t):
        starts.append(start)
        dt = rand_around_time()
        ends.append(start + dt)
    ends.sort()
    return starts, ends


def events_to_cumulative_samples(events, sample_period, min_t, max_t):
    n = 0
    sample_index = 0
    n_counts = int(np.ceil((max_t - min_t) / sample_period))
    counts = np.zeros(n_counts, dtype=int)
    while n < len(events):
        next_sample_index = (events[n] - min_t) // sample_period
        while sample_index < next_sample_index:
            if sample_index >= n_counts:
                break
            counts[sample_index] = n
            sample_index += 1
        n += 1
    return counts



def get_sim_cum_starts_ends(sample_period, min_t, max_t, sim_start_t=0):
    starts, ends = get_simulation_events(sim_start_t, max_t)
    starts = [t for t in starts if t >= min_t]
    ends = [t for t in ends if t >= min_t]
    starts_samples = events_to_cumulative_samples(starts, sample_period, min_t, max_t)
    ends_samples = events_to_cumulative_samples(ends, sample_period, min_t, max_t)
    return starts_samples, ends_samples


def get_simulation(sample_period):
    counts = np.zeros(int(np.ceil(1440 / sample_period)))
    starts, ends = get_simulation_events()
    events = [{'type': 'start', 'val': x} for x in starts] + [{'type': 'end', 'val': x} for x in ends]
    events.sort(key=lambda x: x['val'], reverse=True)
    curr_n = 0
    curr_index = 0
    while len(events) != 0:
        event = events.pop()
        next_index = event['val'] // sample_period
        while curr_index < next_index:
            if curr_index >= len(counts):
                break
            counts[curr_index] = curr_n
            curr_index += 1

            # this is from our wrong assumption
            # curr_n -= K_PARAM * curr_n

        if event['type'] == 'start':
            curr_n += 1
        else:
            curr_n -= 1

    return counts


def main():
    # t0 = time()

    # counts = []
    # for _ in range(10):
    #     n = 0
    #     for x in my_random_times(0, 1440):
    #         n += 1
    #     counts.append(n)
    # print(counts)
    # print(np.mean(counts))

    # print(time() - t0)


    # n_sims = 100
    # sample_period = 1
    # sim = sum(get_simulation(sample_period) for _ in range(n_sims)) / n_sims
    # for i, y in enumerate(sim):
    #     print(f"{i*sample_period}\t{y}")


    # for i in range(1440):
    #     print(f"{i}\t{my_λ(i - 0.5/K_PARAM) / K_PARAM}")  # this is what really happens!!!

    # for i in range(1440):
    #     print(f"{i}\t{my_n(i, 0)}")


    # Normaldistribution(µ=747.121, s=28.0257703338)
    # my_λ(x - 0.5/K_PARAM) / K_PARAM ~ my_n(x, 0)

    # AVG_AROUND_TIME * my_λ(t - 0.32 * AVG_AROUND_TIME) ~ n(t)


    # times = rand_around_time(747000)
    # plt.hist(times, bins='auto')
    # print(np.mean(times))
    # plt.show()



    # starts, ends = get_simulation_events()

    # starts = [t for t in starts if t > 1000]
    # ends = [t for t in ends if t > 1000]

    # for t, n in zip(starts, range(1, len(starts) + 1)):
    #     print(f"{t}\t{n}")

    # print("--DONE--")
    # for t, n in zip(ends, range(1, len(ends) + 1)):
    #     print(f"{t}\t{n}")




    starts_samples, ends_samples = get_sim_cum_starts_ends(1, 0, 1440)
    # plt.plot(1 - np.exp(-(starts_samples)))
    # plt.plot(starts_samples - ends_samples)

    plt.scatter(starts_samples, ends_samples)
    plt.plot(starts_samples, starts_samples, 'r')
    plt.show()







if __name__ == '__main__':
    main()

# https://pypi.org/project/pynverse/
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0023299
# https://bmcgeriatr.biomedcentral.com/articles/10.1186/s12877-016-0201-x#Sec12
# https://www.healthline.com/health/average-mile-time#by-age
# http://www.pace-calculator.com/5k-pace-comparison.php