from simulation_2 import get_simple_sim
import matplotlib.pyplot as plt
import numpy as np
from tictoc import *
import cProfile

'''Simple simulation'''


DISTANCE = 1
DAY_LENGTH = 200


def get_sim(rate, duration, day_length):
    mile_time = duration / DISTANCE
    return get_simple_sim(DISTANCE, day_length, rate, mile_time,
        n_times_around=1, same_direction_prop=1)


def get_counts(rate, duration, n_trials):

    # tic()
    # sim = get_sim(rate, duration, 2 * duration)
    # t = 200
    # counts = []
    # for i in range(n_trials):
    #     counts.append(sim.n_people(t))
    #     sim.refresh_pedestrians()
    # toc()

    tic()
    sample_period = duration
    t0 = 2 * duration
    day_length = t0 + sample_period * (n_trials - 1)
    sim = get_sim(rate, duration, day_length)
    counts = []
    t = t0
    for i in range(n_trials):
        counts.append(sim.n_people(t))
        t += sample_period
    toc()

    return counts


def plot_number_of_people_over_time(rate, duration, n_trials):
    sim = get_sim(rate, duration, DAY_LENGTH)

    time_samples = np.arange(0, DAY_LENGTH, 1)
    for i in range(n_trials):
        people_counts = [sim.n_people(t) for t in time_samples]
        plt.plot(time_samples, people_counts)
        sim.refresh_pedestrians()
    plt.plot(time_samples, np.fmin(rate * time_samples, rate * duration), label='expected number of people', color='k')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('people')
    plt.title(f'number of people over time, {n_trials} simulations\nλ={rate}, τ={duration}')
    plt.show()


def show_n_distribution(rate, duration, n_trials):
    counts = get_counts(rate, duration, n_trials)

    mean = np.mean(counts)
    sd = np.std(counts, ddof=1)

    print(f"Mean: {mean}")
    print(f"SD: {sd}")
    print(f"Var: {sd**2}")
    print(f"N: {n_trials}")

    plt.hist(counts, bins=range(min(counts), max(counts) + 1, 1), density=True, label='histogram')

    time_samples = np.arange(0, DAY_LENGTH, 1)
    pdf = np.exp(-0.5 * ((time_samples - mean) / sd)**2)/(np.sqrt(2 * np.pi) * sd)
    plt.plot(time_samples, pdf, label='normal distribution fit')

    plt.xlabel('number of people')
    plt.ylabel('probability density')
    plt.title(f'Number of people\nλ={rate}, τ={duration}, µ={mean:.2f}, σ={sd:.2f}, n={n_trials}')

    plt.legend()

    plt.show()


def get_stds(rates, durations, n_trials):
    stds = []
    for rate, duration in zip(rates, durations):
        counts = get_counts(rate, duration, n_trials)
        sd = np.std(counts, ddof=1)
        stds.append(sd)
    return np.array(stds)


def plot_std_vs_rate(duration, min_rate, max_rate, n_samples, n_trials):
    durations = np.ones(n_samples) * duration
    rates = np.linspace(min_rate, max_rate, n_samples)
    stds = get_stds(rates, durations, n_trials)
    plt.plot(rates, stds**2)
    plt.xlabel('rate')
    plt.ylabel('std')
    plt.title(f'std vs rate, duration = {duration}')
    plt.show()


# plot_std_vs_rate(4, 2, 8, 5, 100)


# cProfile.runctx('plot_std_vs_rate(4, 2, 8, 5, 100)', globals(), locals(), sort='cumulative')


# plot_number_of_people_over_time(rate=2, duration=60, n_trials=3)

show_n_distribution(rate=0.5, duration=5, n_trials=10000)


# IT FOLLOWS A POISSON DISTRIBUTION!!


# 10.93306670341633

