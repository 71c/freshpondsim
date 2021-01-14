import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from tqdm import tqdm

def get_random_velocities_and_distances_func(random_duration_func, distance_traveled):
    def ret(n):
        durations = random_duration_func(n)
        distances = np.ones(n) * distance_traveled
        velocities = distances / durations
        return np.array([velocities, distances]).T
    return ret


def random_exponential_func(scale):
    def ret(n):
        return np.random.exponential(scale=scale, size=n)
    return ret


def get_exponential_rand_velocities_and_distances_func(scale):
    return get_random_velocities_and_distances_func(random_exponential_func(scale), 1)


def entrance_rate_func(t):
    # return 5 + 2 * np.sin(2*np.pi * t/30.0)
    return 5


def entrance_rate_integral(t):
    return 5 * t


start_time = 0
end_time = 200

sim = FreshPondSim(distance=1,
                   start_time=start_time,
                   end_time=end_time,
                   entrances=[0],
                   entrance_weights=[1],
                   rand_velocities_and_distances_func=get_exponential_rand_velocities_and_distances_func(5),
                   entrance_rate=entrance_rate_func,
                   entrance_rate_integral=entrance_rate_integral,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)

a = 50
samples = []
true_means = []
# estimated_variances = []
# true_variances = []
for _ in tqdm(range(50)):
    # estimated_n_integral = (end_time - a) * np.mean([sim.n_people(x) for x in np.random.uniform(a, end_time, 50000)])
    # samples.append(estimated_n_integral / sim.num_entrances_in_interval(a, end_time))
    # times = [x.end_time - x.start_time for x in sim.get_pedestrians_in_interval(a, end_time)]
    # print(len(times))
    # true_means.append(np.mean(times))
    # sim.refresh_pedestrians()

    pedestrians = sim.get_pedestrians_in_interval(a, end_time)
    time_lengths = [x.end_time - x.start_time for x in pedestrians]

    # print(sim.pedestrians)

    # start_times = [x.start_time for x in pedestrians]
    # end_times = [x.end_time for x in pedestrians]

    start_times = [x.start_time for x in sim.pedestrians if a <= x.start_time <= end_time]
    end_times = [x.end_time for x in sim.pedestrians if a <= x.end_time <= end_time]

    # samples.append((sum(end_times) - sum(start_times)) / len(start_times))
    # samples.append(np.mean(end_times) - np.mean(start_times))

    # print(start_times, end_times)

    m = sim.n_people(a) * (end_time - a)
    for u in start_times:
        m += end_time - u
    for b in end_times:
        m -= end_time - b

    m2 = sim.n_people(a) * (end_time - a) + (len(start_times) - len(end_times)) * end_time + sum(end_times) - sum(start_times)

    assert np.isclose(m, m2)

    # estimated_n_integral = (end_time - a) * np.mean([sim.n_people(x) for x in np.random.uniform(a, end_time, 100000)])
    # print(m, estimated_n_integral)
    # assert sim.num_entrances_in_interval(a, end_time) == len(start_times)

    # U = sim.n_people(a) / (len(pedestrians) / (end_time - a))
    # V = m / ((len(start_times)))
    # print(U, V, np.mean(time_lengths))

    samples.append(m / ((len(start_times))))
    # samples.append(sim.n_people(a) / (len(pedestrians) / (end_time - a)))

    true_means.append(np.mean(time_lengths))

    # estimated_variances.append(np.var(end_times, ddof=1) - np.var(start_times, ddof=1))
    # true_variances.append(np.var(time_lengths, ddof=1))

    sim.refresh_pedestrians()

# print(np.var(true_means, ddof=1), np.var(samples, ddof=1))

print(ttest_rel(true_means, samples))
print(ttest_1samp(true_means, 5))
print(ttest_1samp(samples, 5))

print(np.mean(true_means), np.mean(samples))
print(np.mean(true_means) - np.mean(samples))

# print(true_means)
# print(samples)


# sample_interval = 3
# estimated_entrance_rates = []
# people_counts = []

# t = start_time
# while t < end_time:
#     t += sample_interval
#     estimated_entrance_rates.append(sim.num_entrances_in_interval(t - sample_interval, t) / 1)
#     # estimated_entrance_rates.append(entrance_rate_func(t) * sample_interval)
#
#     people_counts.append(np.mean([sim.n_people(u) for u in np.linspace(t - sample_interval, t, 50)]))
#
# # print(estimated_entrance_rates)
# print(np.sum(people_counts) * sample_interval / np.sum(estimated_entrance_rates))
#
# # people_counts.extend([0 for _ in range(len(people_counts) - 1)])
# people_counts.extend([0 for _ in range(3)])
#
#
# # print(estimated_entrance_rates)
# # print(people_counts)
#
# original = estimated_entrance_rates
# convolved = people_counts
#
# recovered, remainder = signal.deconvolve(convolved, original)
# print(recovered)
# # print(list(zip(signal.convolve(estimated_entrance_rates, recovered), people_counts)))
