import numpy as np
import scipy.stats
from freshpondsim import random_times
from tqdm import tqdm
from inout_theory import InOutTheory
import matplotlib.pyplot as plt


def estimate_mean_duration(start_n_people, entrance_times, exit_times, t1, t2):
    n_integral = start_n_people * (t2 - t1) + np.sum(t2 - entrance_times) - np.sum(t2 - exit_times)
    estimated_mean_time = n_integral / ((len(entrance_times) + len(exit_times))/2)
    return estimated_mean_time


# Set up entrance rate
# Sinusoidal entrance rate
entrance_rate_constant = 2
a = 0.95 * entrance_rate_constant
period = 20
freq = 1/period
omega = 2*np.pi * freq
def rate(t):
    if t < 0:
        return 0.0
    return entrance_rate_constant + a * np.cos(omega * t)
def cum_rate(t):
    if t < 0:
        return 0.0
    return entrance_rate_constant * t + a / omega * np.sin(omega * t)
cum_rate_inverse = None

# Set up duration distribution
scale = 4.0  # scale parameter of Weibull distribution
k = 1.5  # shape parameter of weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)

iot = InOutTheory(duration_dist, rate, cum_rate, cum_rate_inverse)


# This is when no one enters anymore
t_sim_end = 100

# define sampling interval
t_sample_start = 85
t_sample_end = t_sim_end

n_simulations = 10_000
est_mean_durations = []
true_mean_durations = []
for i in tqdm(range(n_simulations)):
    entrance_times, exit_times = iot.sample_realization(t_sim_end)

    n_entrances_before = len(entrance_times[entrance_times < t_sample_start])
    n_exits_before = len(exit_times[exit_times < t_sample_start])
    n_people_at_start = n_entrances_before - n_exits_before
    sample_entrance_times = entrance_times[(t_sample_start <= entrance_times) & (entrance_times <= t_sample_end)]
    sample_exit_times = exit_times[(t_sample_start <= exit_times) & (exit_times <= t_sample_end)]

    est_mean_time = estimate_mean_duration(n_people_at_start, sample_entrance_times, sample_exit_times, t_sample_start, t_sample_end)
    est_mean_durations.append(est_mean_time)

    true_mean_duration = (exit_times - entrance_times).mean()
    true_mean_durations.append(true_mean_duration)


n_integral = iot.n_integral(t_sample_start, t_sample_end)
n_entrance = iot.expected_entrances_in_interval(t_sample_start, t_sample_end)
n_exit = iot.cumulative_exit_rate(t_sample_end) - iot.cumulative_exit_rate(t_sample_start)
expected_estimated_mean_duration = n_integral / ((n_entrance + n_exit) / 2)

print('Duration expected value:', duration_dist.mean())
print('Average true mean duration:', np.mean(true_mean_durations))
print('Average estimated mean duration:', np.mean(est_mean_durations))
print('Expected estimated mean duration:', expected_estimated_mean_duration)


plt.hist(est_mean_durations, bins='auto')
plt.show()

