'''
Compares different methods of approximating average duration when the
correspondence between entrances and exits is not known
'''
import numpy as np
import scipy.stats
from freshpondsim import random_times
from tqdm import tqdm
from inout_theory import InOutTheory
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma
from scipy import integrate
from tictoc import *


VERSIONS_UNINFORMED = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
VERSIONS = VERSIONS_UNINFORMED + ['informed']


def get_mean_duration_estimation_constants_weibull(est_mu, dt, k=1.5):
    G1 = Gamma(1 + 1/k)
    G2 = Gamma(1 + 2/k)
    G3 = Gamma(1 + 3/k)
    s = est_mu / G1
    gamma = G2 / (2 * G1)
    VA = s**2 * (G3 / (3 * G1) - gamma**2)

    def pdf_A(x):
        return np.exp(-(x/s)**k) / est_mu
    prob_exceed, abserr = integrate.quad(pdf_A, dt, np.inf)

    VT = s**2 * (G2 - G1**2)

    return {'gamma': gamma, 'VA': VA, 'prob_exceed': prob_exceed, 'VT': VT}


def estimate_mean_duration_versions(start_n_people, entrance_times, exit_times, t1, t2):
    E = len(exit_times)
    Lambda = len(entrance_times)

    n_integral = start_n_people * (t2 - t1) + np.sum(t2 - entrance_times) - np.sum(t2 - exit_times)
    estimated_mean_time_v1 = n_integral / ((Lambda + E)/2)
    estimated_mean_time_v2 = 0.5 * (n_integral / Lambda + n_integral / E)
    estimated_mean_time_v3 = n_integral / E
    estimated_mean_time_v4 = n_integral / Lambda

    # estimated_mean_time_v5 = n_integral / (0.7 * E + 0.3 * Lambda)
    estimated_mean_time_v5 = ((n_integral / (0.7 * E + 0.3 * Lambda)) + (n_integral / (0.3 * E + 0.7 * Lambda))) / 2

    tmp = E/Lambda - 1

    # http://www.columbia.edu/~ww2040/LL_OR.pdf
    K = 1.0
    estimated_mean_time_v6 = estimated_mean_time_v4 * (1 - tmp * K)

    # Wrong calculation - forgot covariance
    # n2 = start_n_people + Lambda - E
    # d = get_mean_duration_estimation_constants_weibull(estimated_mean_time_v4, k=1.5)
    # gamma = d['gamma']
    # VA = d['VA']
    # tmp2 = (start_n_people + n2) * VA / Lambda
    # tmp3 = n_integral / (n_integral + tmp2)
    # tmp4 = tmp3 * tmp * gamma
    # estimated_mean_time_v7 = (1 - tmp4) * estimated_mean_time_v4

    # Correct calculation
    mu0 = estimated_mean_time_v4
    if tmp == 0:
        estimated_mean_time_v7 = mu0
    else:
        n1 = start_n_people
        n2 = n1 + Lambda - E
        d = get_mean_duration_estimation_constants_weibull(mu0, dt=t2-t1, k=1.5)
        gamma = d['gamma']
        VA = d['VA']
        prob_exceed = d['prob_exceed']
        VT = d['VT']
        C1 = tmp * gamma
        # print(2 * n1 * VT * prob_exceed, 2 * n1 * VT * prob_exceed / ((n1 + n2) * VA - 2 * n1 * VT * prob_exceed))
        # C2 = (n1 + n2) * VA / Lambda**2
        C2 = ((n1 + n2) * VA - 2 * n1 * VT * prob_exceed) / Lambda**2
        cov_hat = C1 * C2
        var_hat = C1**2 * C2
        bias_hat = C1 * mu0
        
        alpha_hat = (bias_hat**2 + cov_hat) / (bias_hat**2 + var_hat)
        # print(alpha_hat)
        # print(tmp, gamma, bias_hat, cov_hat, var_hat, alpha_hat)
        # if alpha_hat < 0.0:
        #     alpha_hat = 0
        # if alpha_hat > 1.0:
        #     alpha_hat = 1.0

        estimated_mean_time_v7 = mu0 - alpha_hat * bias_hat

    return {
        'v1': estimated_mean_time_v1,
        'v2': estimated_mean_time_v2,
        'v3': estimated_mean_time_v3,
        'v4': estimated_mean_time_v4,

        'v5': estimated_mean_time_v5,
        'v6': estimated_mean_time_v6,
        'v7': estimated_mean_time_v7
    }


def get_mean_duration_estimations_uninformed(iot, t_sample_start, t_sample_end, n_simulations, do_tqdm=True):
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


t_sample_start = 100.0
n_sample_ends = 4
max_sample_end = 115.0
t_sample_ends = np.linspace(105.0, max_sample_end, num=n_sample_ends)
n_simulations = 900

biases_wrt_mu = {v: [] for v in VERSIONS}
rmses_wrt_mu = {v: [] for v in VERSIONS}
rmses_wrt_mulambda = {v: [] for v in VERSIONS}

# Naive method time complexity: n_ends * n_simulations * O(t_end)
# Smart method time complexity: n_simulations * O(t_end)

# New version
tic()

# The i'th element of `estimations` corresponds to estimations
# for t_sample_end being t_sample_ends[i].
estimations = [
    {v: np.empty(n_simulations) for v in VERSIONS}
    for _ in range(n_sample_ends)]

for simulation_num in range(n_simulations):
    entrance_times, exit_times = iot.sample_realization(max_sample_end)

    n_entrances_before = len(entrance_times[entrance_times <= t_sample_start])
    n_exits_before = len(exit_times[exit_times <= t_sample_start])
    n_people_at_start = n_entrances_before - n_exits_before
    
    for i, t_sample_end in enumerate(t_sample_ends):
        sample_entrance_times = entrance_times[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)]
        sample_exit_times = exit_times[(t_sample_start < exit_times) & (exit_times <= t_sample_end)]

        versions = estimate_mean_duration_versions(n_people_at_start, sample_entrance_times, sample_exit_times, t_sample_start, t_sample_end)
        for v in VERSIONS_UNINFORMED:
            estimations[i][v][simulation_num] = versions[v]
        version_informed = np.mean((exit_times - entrance_times)[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)])
        estimations[i]['informed'][simulation_num] = version_informed

for i, t_sample_end in enumerate(t_sample_ends):
    est_means = estimations[i]
    for name in VERSIONS:
        samples = est_means[name]
        avg = np.mean(samples)
        bias_wrt_mu = avg - iot._mean_duration
        rmse_wrt_mu = np.sqrt(np.mean((samples - iot._mean_duration)**2))
        rmse_wrt_mulambda = np.sqrt(np.mean((samples - est_means['informed'])**2))
        biases_wrt_mu[name].append(bias_wrt_mu)
        rmses_wrt_mu[name].append(rmse_wrt_mu)
        rmses_wrt_mulambda[name].append(rmse_wrt_mulambda)

toc()

# Old, inefficient version
# tic()
# for t_sample_end in tqdm(t_sample_ends):
#     est_means = get_mean_duration_estimations_uninformed(iot, t_sample_start, t_sample_end, n_simulations, do_tqdm=False)
#     for name in VERSIONS:
#         samples = est_means[name]
#         avg = np.mean(samples)
#         bias_wrt_mu = avg - iot._mean_duration
#         rmse_wrt_mu = np.sqrt(np.mean((samples - iot._mean_duration)**2))
#         rmse_wrt_mulambda = np.sqrt(np.mean((samples - est_means['informed'])**2))
#         biases_wrt_mu[name].append(bias_wrt_mu)
#         rmses_wrt_mu[name].append(rmse_wrt_mu)
#         rmses_wrt_mulambda[name].append(rmse_wrt_mulambda)
# toc()

for v in VERSIONS:
    plt.plot(t_sample_ends, rmses_wrt_mulambda[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu^\lambda$')
plt.legend()

plt.figure()
for v in VERSIONS:
    plt.plot(t_sample_ends, rmses_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu$')
plt.legend()

plt.figure()
for v in VERSIONS:
    plt.plot(t_sample_ends, biases_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('bias w.r.t. $\mu$')
plt.legend()

plt.show()
