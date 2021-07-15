import numpy as np
import scipy.stats
from freshpondsim import random_times
from tqdm import tqdm
from inout_theory import InOutTheory
import matplotlib.pyplot as plt


VERSIONS_UNINFORMED = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
VERSIONS = VERSIONS_UNINFORMED + ['informed']


def estimate_mean_duration_versions(start_n_people, entrance_times, exit_times, t1, t2):
    n_integral = start_n_people * (t2 - t1) + np.sum(t2 - entrance_times) - np.sum(t2 - exit_times)
    estimated_mean_time_v1 = n_integral / ((len(entrance_times) + len(exit_times))/2)
    estimated_mean_time_v2 = 0.5 * (n_integral / len(entrance_times) + n_integral / len(exit_times))
    estimated_mean_time_v3 = n_integral / len(exit_times)
    estimated_mean_time_v4 = n_integral / len(entrance_times)

    estimated_mean_time_v5 = n_integral / (0.7 * len(exit_times) + 0.3 * len(entrance_times))

    # http://www.columbia.edu/~ww2040/LL_OR.pdf
    tmp = (len(exit_times)/len(entrance_times) - 1)
    K = 1.0
    estimated_mean_time_v6 = estimated_mean_time_v4 * (1 - tmp * K)

    # v5 and v6 are only appropriate when start time is 0

    # min_exit_time, max_exit_time = min(exit_times), max(exit_times)
    # min_entrance_time, max_entrance_time = min(entrance_times), max(entrance_times)

    # if max_entrance_time > max_exit_time:
    #     # t2 = max_exit_time
    #     t2 = min(entrance_times[entrance_times > max_exit_time])
    #     entrance_times = entrance_times[entrance_times < max_exit_time]

    # if min_exit_time < min_entrance_time:
    #     exit_times_before = exit_times[exit_times < min_entrance_time]
    #     # t1 = min_entrance_time
    #     t1 = max(exit_times_before)
    #     start_n_people = start_n_people - len(exit_times_before)
    #     exit_times = exit_times[exit_times > min_entrance_time]
    
    # n_integral = start_n_people * (t2 - t1) + np.sum(t2 - entrance_times) - np.sum(t2 - exit_times)
    # estimated_mean_time_v5 = 0.5 * (n_integral / len(entrance_times) + n_integral / len(exit_times))
    # estimated_mean_time_v6 = n_integral / ((len(entrance_times) + len(exit_times))/2)

    return {
        'v1': estimated_mean_time_v1,
        'v2': estimated_mean_time_v2,
        'v3': estimated_mean_time_v3,
        'v4': estimated_mean_time_v4,

        'v5': estimated_mean_time_v5,
        'v6': estimated_mean_time_v6
    }


def get_mean_duration_estimations(iot, t_sample_start, t_sample_end, n_simulations, do_tqdm=True):
    est_means = {v: np.empty(n_simulations) for v in VERSIONS}

    it = range(n_simulations)
    if do_tqdm:
        it = tqdm(it)
    for i in it:
        # it's ok to do this because all that matters is what happens before
        # time t_sample_end
        entrance_times, exit_times = iot.sample_realization(t_sample_end)

        n_entrances_before = len(entrance_times[entrance_times < t_sample_start])
        n_exits_before = len(exit_times[exit_times < t_sample_start])
        n_people_at_start = n_entrances_before - n_exits_before
        sample_entrance_times = entrance_times[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)]
        sample_exit_times = exit_times[(t_sample_start < exit_times) & (exit_times <= t_sample_end)]

        versions = estimate_mean_duration_versions(n_people_at_start, sample_entrance_times, sample_exit_times, t_sample_start, t_sample_end)
        for v in VERSIONS_UNINFORMED:
            est_means[v][i] = versions[v]
        
        version_informed = np.mean((exit_times - entrance_times)[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)])
        est_means['informed'][i] = version_informed

    return est_means


#     n_integral = iot.n_integral(t_sample_start, t_sample_end)
#     n_entrance = iot.expected_entrances_in_interval(t_sample_start, t_sample_end)
#     n_exit = iot.cumulative_exit_rate(t_sample_end) - iot.cumulative_exit_rate(t_sample_start)
#     expected_estimated_mean_duration_v1 = n_integral / ((n_entrance + n_exit) / 2)
#     expected_estimated_mean_duration_v2 = 0.5 * (n_integral / n_entrance + n_integral / n_exit)
#     expected_estimated_mean_duration_v3 = n_integral / n_exit
#     expected_estimated_mean_duration_v4 = n_integral / n_entrance
    




entrance_rate_constant = 15.0

# Set up entrance rate
# Sinusoidal entrance rate
# a = 0.95 * entrance_rate_constant
# period = 20
# freq = 1/period
# omega = 2*np.pi * freq
# def rate(t):
#     if t < 0:
#         return 0.0
#     return entrance_rate_constant + a * np.cos(omega * t)
# def cum_rate(t):
#     if t < 0:
#         return 0.0
#     return entrance_rate_constant * t + a / omega * np.sin(omega * t)
# cum_rate_inverse = None


###### Constant Entry Rate
def rate(t):
    if t < 0:
        return 0
    return entrance_rate_constant
def cum_rate(t):
    if t < 0:
        return 0
    return entrance_rate_constant * t
def cum_rate_inverse(y):
    assert y >= 0
    return y / entrance_rate_constant


# Set up duration distribution
scale = 4.0  # scale parameter of Weibull distribution
k = 1.5  # shape parameter of weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)

iot = InOutTheory(duration_dist, rate, cum_rate, cum_rate_inverse)



# t_sample_start = 0.0
# t_sample_end = 10.0
# n_simulations = 2500
# est_means = get_mean_duration_estimations(iot, t_sample_start, t_sample_end, n_simulations)
# true_mean_duration = iot._mean_duration
# print(f'True mean duration: {true_mean_duration}')
# # for name, samples in [('v1', est_means_v1), ('v2', est_means_v2), ('v3', est_means_v3), ('v4', est_means_v4)]:
# for name in VERSIONS:
#     samples = est_means[name]
#     avg = np.mean(samples)
#     bias = avg - true_mean_duration
#     rmse = np.sqrt(np.mean((samples - true_mean_duration)**2))
#     print(f"{name}: Average: {avg:.4g}, Bias: {bias:.3g}, RMSE: {rmse:.4g}")

#     plt.figure()
#     plt.hist(samples, bins='auto')
#     plt.title(name)
# plt.show()



t_sample_start = 30.0
t_sample_ends = np.linspace(35.0, 70.0, num=10)
n_simulations = 1000

biases = {v: [] for v in VERSIONS}
rmses = {v: [] for v in VERSIONS}

for t_sample_end in tqdm(t_sample_ends):
    est_means = get_mean_duration_estimations(iot, t_sample_start, t_sample_end, n_simulations, do_tqdm=False)
    for name in VERSIONS:
        samples = est_means[name]
        avg = np.mean(samples)
        bias = avg - iot._mean_duration
        rmse = np.sqrt(np.mean((samples - iot._mean_duration)**2))
        biases[name].append(bias)
        rmses[name].append(rmse)

for v in VERSIONS:
    plt.plot(t_sample_ends, rmses[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE')
plt.legend()

plt.figure()
for v in VERSIONS:
    plt.plot(t_sample_ends, biases[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('bias')
plt.legend()

plt.show()
