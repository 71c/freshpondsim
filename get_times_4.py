'''
Compares different methods of approximating average duration when the
correspondence between entrances and exits is not known
'''
import numpy as np
import scipy.stats
from freshpondsim import random_times
from tqdm import tqdm, trange
from inout_theory import InOutTheory
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma
from scipy.special import gammaincc
from scipy import integrate
from tictoc import *
from cycler import cycler


# VERSIONS_UNINFORMED = ['v1', 'v2', 'v3', 'v4', 'v5', 'bias corrected Lambda', 'bias corrected Lambda minimum MSE', 'bias corrected E', 'avg bias corrected E, Lambda', 'bias corrected Lambda 2', 'bias corrected Lambda minimum MSE 2']
# VERSIONS_UNINFORMED = ['v4', 'bias corrected Lambda', 'bias corrected Lambda minimum MSE', 'bias corrected E', 'avg bias corrected E, Lambda', 'bias corrected Lambda 2', 'bias corrected Lambda minimum MSE 2 a', 'bias corrected Lambda minimum MSE 2 a no cov', 'E minimum MSE', 'avg Lambda, E minimum MSE', 'minimum MSE direct'] # , 'bias corrected Lambda minimum MSE 2 b', 'bias corrected Lambda minimum MSE 2 c'
VERSIONS_UNINFORMED = ['v1', 'v2', 'v3', 'v4', 'v5', 'bias corrected Lambda',  'bias corrected E', 'bias corrected Lambda 2', 'bias corrected Lambda minimum MSE 2 a no cov', 'E minimum MSE', 'avg Lambda, E minimum MSE', 'MLE']
VERSIONS_INFORMED = ['entrances', 'exits', 'entrances and exits weighted average', 'entrances and exits unweighted average']#, 'union'
VERSIONS = VERSIONS_UNINFORMED + VERSIONS_INFORMED


def get_mean_duration_estimation_constants_weibull(est_mu, dt, k=1.5):
    G1 = Gamma(1 + 1/k)
    G2 = Gamma(1 + 2/k)
    G3 = Gamma(1 + 3/k)
    s = est_mu / G1
    gamma = G2 / (2 * G1)
    VA = s**2 * (G3 / (3 * G1) - gamma**2)

    # def pdf_A(x):
    #     return np.exp(-(x/s)**k) / est_mu
    # prob_exceed, abserr = integrate.quad(pdf_A, dt, np.inf)

    a = 1/k
    z = (dt/s)**k
    prob_exceed = s/k * Gamma(a) * gammaincc(a, z) / est_mu

    VT = s**2 * (G2 - G1**2)

    return {'gamma': gamma, 'VA': VA, 'prob_exceed': prob_exceed, 'VT': VT}


def estimate_mean_duration_MLE_and_var(start_n_people, entrance_times, exit_times, t1, t2):
    assert np.all((t1 < entrance_times) & (entrance_times <= t2))
    assert np.all((t1 < exit_times) & (exit_times <= t2))

    E = len(exit_times)
    Lambda = len(entrance_times)

    n_integral = start_n_people * (t2 - t1) + np.sum(t2 - entrance_times) - np.sum(t2 - exit_times)

    n1 = start_n_people
    n2 = n1 + Lambda - E

    tmp = E/Lambda - 1

    k_hat = 1.5

    mu0 = n_integral / Lambda

    d = get_mean_duration_estimation_constants_weibull(mu0, dt=t2-t1, k=k_hat)
    gamma = d['gamma']

    K = tmp * gamma
    estimated_mean_time_Lambda_MLE = mu0 / (1 + K)

    d = get_mean_duration_estimation_constants_weibull(estimated_mean_time_Lambda_MLE, dt=t2-t1, k=k_hat)
    VA = d['VA']
    VT = d['VT']
    prob_exceed = d['prob_exceed']

    # var_mu0 = (n1 + n2) * VA / Lambda**2
    var_mu0 = ((n1 + n2) * VA - 2 * VT * n1 * prob_exceed) / Lambda**2
    # var_mu0 = ((2 * Lambda/(t2-t1) * mu0) * VA - 2 * VT * Lambda/(t2-t1) * mu0 * prob_exceed) / Lambda**2
    

    # print(var_mu0 / (var_mu0 + VT / Lambda))

    # estimated_variance = (var_mu0 + VT / Lambda) / (1 + K)**2
    # # estimated_variance = VT / Lambda
    # return estimated_mean_time_Lambda_MLE, estimated_variance

    # denom = Lambda * (1 + K)
    # estimated_variance = Lambda * VT / denom**2 + (n1 + n2) * VA / denom**2
    # return estimated_mean_time_Lambda_MLE, estimated_variance


    # d = get_mean_duration_estimation_constants_weibull(mu0, dt=t2-t1, k=k_hat)
    # VA = d['VA']
    # VT = d['VT']

    # var_mu0 = (n1 + n2) * VA / Lambda**2

    estimated_variance = VT / Lambda + var_mu0
    return mu0, estimated_variance



entrance_rate_constant = 15.0

# Set up entrance rate
###### Sinusoidal entrance rate
# a = 0.7 * entrance_rate_constant
# period = 20.0
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


##### Constant Entry Rate
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


##### Linear Entry Rate
# t_st = 100.0
# lambda0 = 66.0
# beta = 0.5
# def rate(t):
#     if t < 0:
#         return 0
#     return lambda0 + beta * (t - t_st)
# def cum_rate(t):
#     if t < 0:
#         return 0
#     return (lambda0 - t_st * beta) * t + 0.5 * beta * t**2
# def cum_rate_inverse(y):
#     return (beta * t_st - lambda0 + np.sqrt(2 * y * beta + (lambda0 - t_st * beta)**2)) / beta


# Set up duration distribution
scale = 4.0  # scale parameter of Weibull distribution
k = 1.5  # shape parameter of weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)

iot = InOutTheory(duration_dist, rate, cum_rate, cum_rate_inverse)


t_sample_start = 50.0
t_sample_end = 57.0
n_simulations = 3000


means = []
variances = []

for simulation_num in trange(n_simulations):
    entrance_times, exit_times = iot.sample_realization(t_sample_end)

    n_entrances_before = len(entrance_times[entrance_times <= t_sample_start])
    n_exits_before = len(exit_times[exit_times <= t_sample_start])
    n_people_at_start = n_entrances_before - n_exits_before

    sample_entrance_times = entrance_times[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)]
    sample_exit_times = exit_times[(t_sample_start < exit_times) & (exit_times <= t_sample_end)]

    est_mu, est_var = estimate_mean_duration_MLE_and_var(n_people_at_start, sample_entrance_times, sample_exit_times, t_sample_start, t_sample_end)

    means.append(est_mu)
    variances.append(est_var)

print(np.mean(means), np.var(means, ddof=1))
print(np.mean(variances), np.std(variances))

plt.hist(means, bins='auto')
plt.figure()
plt.hist(variances, bins='auto')
plt.show()
