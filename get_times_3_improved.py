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


def estimate_mean_duration_versions_uninformed(start_n_people, entrance_times, exit_times, t1, t2):
    assert np.all((t1 < entrance_times) & (entrance_times <= t2))
    assert np.all((t1 < exit_times) & (exit_times <= t2))

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

    k_hat = 1.5

    mu0 = estimated_mean_time_v4
    d = get_mean_duration_estimation_constants_weibull(mu0, dt=t2-t1, k=k_hat)
    gamma = d['gamma']
    VA = d['VA']
    prob_exceed = d['prob_exceed']
    VT = d['VT']

    # http://www.columbia.edu/~ww2040/LL_OR.pdf
    estimated_mean_time_bias_corrected_Lambda = (1 - tmp * gamma) * mu0
    # print((1 - tmp * gamma), (1 - tmp * gamma) / (1 - (tmp * gamma)**2))
    # estimated_mean_time_bias_corrected_Lambda = (1 - tmp * gamma) / (1 - (tmp * gamma)**2) * mu0

    # estimated_mean_time_bias_corrected_Lambda += (tmp * gamma)**2 * estimated_mean_time_bias_corrected_Lambda

    # Correct calculation
    if tmp == 0:
        estimated_mean_time_bias_corrected_Lambda_minimum_MSE = mu0
        # estimated_mean_time_bias_corrected_Lambda_minimum_MSE = estimated_mean_time_bias_corrected_Lambda
    else:
        # Re-do the calculation with improved estimator
        # d = get_mean_duration_estimation_constants_weibull(estimated_mean_time_bias_corrected_Lambda, dt=t2-t1, k=k_hat)
        # gamma = d['gamma']
        # VA = d['VA']
        # prob_exceed = d['prob_exceed']
        # VT = d['VT']

        n1 = start_n_people
        n2 = n1 + Lambda - E
        C1 = tmp * gamma
        # C2 = (n1 + n2) * VA / Lambda**2
        C2 = ((n1 + n2) * VA - 2 * n1 * VT * prob_exceed) / Lambda**2
        cov_hat = C1 * C2
        var_hat = C1**2 * C2
        bias_hat = C1 * mu0
        # bias_hat = C1 * estimated_mean_time_bias_corrected_Lambda
        
        alpha_hat = (bias_hat**2 + cov_hat) / (bias_hat**2 + var_hat)
        # print(alpha_hat)
        if alpha_hat > 1.0:
            alpha_hat = 1.0

        estimated_mean_time_bias_corrected_Lambda_minimum_MSE = mu0 - alpha_hat * bias_hat

    mu1 = estimated_mean_time_v3
    tmp2 = Lambda/E - 1
    estimated_mean_time_E = (1 - tmp2 * gamma) * mu1
    # estimated_mean_time_E = (1 - tmp2 * gamma) / (1 - (tmp2 * gamma)**2) * mu1

    estimated_mean_bias_corrected_E_Lambda = (estimated_mean_time_bias_corrected_Lambda + estimated_mean_time_E) / 2

    # K_lambda = 1 - tmp * gamma
    # K_e = 1 - tmp2 * gamma
    K_lambda = (1 - tmp * gamma) / (1 - (tmp * gamma)**2)
    K_e = (1 - tmp2 * gamma) / (1 - (tmp2 * gamma)**2)
    A = K_lambda**2 / Lambda**2
    B = K_e**2 / E**2
    C = K_lambda * K_e / (Lambda * E)
    D = A + B - 2 * C
    # assert np.isclose((K_lambda/Lambda - K_e/E)**2, D, atol=0, rtol=1e-8)
    # print((K_lambda/Lambda - K_e/E)**2, D)
    assert D >= 0
    # if E == Lambda:
    #     print(E, Lambda, D)

    # print((B - C) / D)
    # print(A, B, C)

    if D != 0:
        alpha = (B - C) / D
        if alpha > 1.0:
            alpha = 1.0
        elif alpha < 0.0:
            alpha = 0.0
        estimated_mean_bias_corrected_E_Lambda = alpha * estimated_mean_time_bias_corrected_Lambda + (1 - alpha) * estimated_mean_time_E

    # print(E * Lambda / (E**2 + Lambda**2))
    # print(f'A: {K_lambda**2 / Lambda**2}, B: {K_e**2 / E**2}, C: {K_lambda * K_e / (Lambda * E)}')

    estimated_mean_time_bias_corrected_Lambda_2 = mu0 / (1 + (tmp * gamma))

    # print((tmp * gamma))



    #### New method of minimum MSE Lambda correction

    ### Version a

    # base_mu = estimated_mean_time_bias_corrected_Lambda_2
    base_mu = mu0 # empirically shown to be best
    # base_mu = estimated_mean_time_bias_corrected_Lambda

    d = get_mean_duration_estimation_constants_weibull(base_mu, dt=t2-t1, k=k_hat)
    gamma = d['gamma']
    VA = d['VA']
    prob_exceed = d['prob_exceed']
    VT = d['VT']

    n1 = start_n_people
    n2 = n1 + Lambda - E
    var_T1_minus_T2 = (n1 + n2) * VA - 2 * VT * n1 * prob_exceed
    var_mu0 = var_T1_minus_T2 / Lambda**2
    K = tmp * gamma
    alpha = base_mu**2 * (1+K) / (base_mu**2 * (1+K)**2 + var_mu0)
    estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a = alpha * mu0


    var_T1_minus_T2 = (n1 + n2) * VA
    var_mu0 = var_T1_minus_T2 / Lambda**2
    alpha = base_mu**2 * (1+K) / (base_mu**2 * (1+K)**2 + var_mu0)
    estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a_no_cov = alpha * mu0


    # ### Version b

    # base_mu = estimated_mean_time_bias_corrected_Lambda_2
    # # base_mu = mu0
    # # base_mu = estimated_mean_time_bias_corrected_Lambda

    # d = get_mean_duration_estimation_constants_weibull(base_mu, dt=t2-t1, k=k_hat)
    # gamma = d['gamma']
    # VA = d['VA']
    # prob_exceed = d['prob_exceed']
    # VT = d['VT']

    # n1 = start_n_people
    # n2 = n1 + Lambda - E
    # var_T1_minus_T2 = (n1 + n2) * VA - 2 * VT * n1 * prob_exceed
    # # print(2 * VT * n1 * prob_exceed / var_T1_minus_T2)
    # var_mu0 = var_T1_minus_T2 / Lambda**2
    # K = tmp * gamma
    # alpha = base_mu**2 * (1+K) / (base_mu**2 * (1+K)**2 + var_mu0)
    # # print(alpha, estimated_mean_time_bias_corrected_Lambda_2 / mu0)
    # # print(alpha / (estimated_mean_time_bias_corrected_Lambda / mu0))
    # estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_b = alpha * mu0

    # ### Version c

    # estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c = estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_b
    # # estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c = mu0

    # for _ in range(5):
    #     dic = get_mean_duration_estimation_constants_weibull(estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c, dt=t2-t1, k=k_hat)
    #     gamma = dic['gamma']
    #     VA = dic['VA']
    #     prob_exceed = dic['prob_exceed']
    #     VT = dic['VT']

    #     var_T1_minus_T2 = (n1 + n2) * VA - 2 * VT * n1 * prob_exceed
    #     var_mu0 = var_T1_minus_T2 / Lambda**2

    #     a = 1+K
    #     b = -mu0
    #     c = var_mu0 / (1+K)
    #     d = b**2 - 4 * a * c
    #     sol1 = (-b + np.sqrt(d)) / (2*a)
    #     sol2 = (-b - np.sqrt(d)) / (2*a)

    #     # print(sol1 - estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c)

    #     estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c = sol1 if abs(sol1 - estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c) < abs(sol2 - estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c) else sol2


    ##### Minimum MSE E correction (new method)
    base_mu = mu1
    d = get_mean_duration_estimation_constants_weibull(base_mu, dt=t2-t1, k=k_hat)
    gamma = d['gamma']
    VA = d['VA']
    prob_exceed = d['prob_exceed']
    VT = d['VT']

    var_mu1 = (n1 + n2) * VA / E**2
    K = tmp2 * gamma
    estimated_mean_time_E_minimum_MSE = mu1 / (1 + K + var_mu1 / (mu1**2 * (1 + K)))

    #### Average of minimum MSEs Lambda, E
    avg_Lambda_E_minimum_MSE = (estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a_no_cov + estimated_mean_time_E_minimum_MSE) / 2


    # ##### Direct MSE
    # C1 = 0.5 # lambda    mu0
    # C2 = 0.5 # e         mu1

    # mu_init = C1 * mu0 + C2 * mu1

    # base_mu = mu_init
    # # base_mu = avg_Lambda_E_minimum_MSE
    # d = get_mean_duration_estimation_constants_weibull(base_mu, dt=t2-t1, k=k_hat)
    # gamma = d['gamma']
    # VA = d['VA']

    # var_mu0 = (n1 + n2) * VA / Lambda**2
    # K_lambda = tmp * gamma

    # var_mu1 = (n1 + n2) * VA / E**2
    # K_e = tmp2 * gamma

    # var_mu_init = C1**2 * var_mu0 + C2**2 * var_mu1
    # # K = C1 * K_lambda + C2 * K_e
    # # K = (C1 * K_lambda * mu0 + C2 * K_e * mu1) / mu_init
    # # K = (C1 * K_lambda * estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a_no_cov + C2 * K_e * estimated_mean_time_E_minimum_MSE) / mu_init
    # K = (C1 * K_lambda * estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a_no_cov + C2 * K_e * estimated_mean_time_E_minimum_MSE) / avg_Lambda_E_minimum_MSE

    # estimated_mean_time_minimum_MSE_direct = mu_init / (1 + K + var_mu_init / (mu_init**2 * (1 + K)))
    # # estimated_mean_time_minimum_MSE_direct = mu_init / (1 + K + var_mu_init / (avg_Lambda_E_minimum_MSE**2 * (1 + K)))


    ##### Direct MSE
    C1 = 0.5 # lambda    mu0
    C2 = 0.5 # e         mu1

    mu_init = C1 * mu0 + C2 * mu1
    
    d = get_mean_duration_estimation_constants_weibull(mu0, dt=t2-t1, k=k_hat)
    gamma = d['gamma']
    VA = d['VA']
    var_mu0 = (n1 + n2) * VA / Lambda**2
    K_lambda = tmp * gamma

    d = get_mean_duration_estimation_constants_weibull(mu1, dt=t2-t1, k=k_hat)
    gamma = d['gamma']
    VA = d['VA']
    var_mu1 = (n1 + n2) * VA / E**2
    K_e = tmp2 * gamma

    var_mu_init = C1**2 * var_mu0 + C2**2 * var_mu1
    # K = C1 * K_lambda + C2 * K_e
    K = (C1 * K_lambda * mu0 + C2 * K_e * mu1) / mu_init

    estimated_mean_time_minimum_MSE_direct = mu_init / (1 + K + var_mu_init / (mu_init**2 * (1 + K)))


    # estimated_mean_time_MLE = mu_init / (1 + C1 * K_lambda + C2 * K_e)
    estimated_mean_time_MLE = mu_init / (1 + K)


    return {
        'v1': estimated_mean_time_v1,
        'v2': estimated_mean_time_v2,
        'v3': estimated_mean_time_v3,
        'v4': estimated_mean_time_v4,

        'v5': estimated_mean_time_v5,
        'bias corrected Lambda': estimated_mean_time_bias_corrected_Lambda,
        'bias corrected Lambda minimum MSE': estimated_mean_time_bias_corrected_Lambda_minimum_MSE,

        'bias corrected E': estimated_mean_time_E,
        'avg bias corrected E, Lambda': estimated_mean_bias_corrected_E_Lambda,

        'bias corrected Lambda 2': estimated_mean_time_bias_corrected_Lambda_2,
        'bias corrected Lambda minimum MSE 2 a': estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a,
        'bias corrected Lambda minimum MSE 2 a no cov': estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_a_no_cov,
        # 'bias corrected Lambda minimum MSE 2 b': estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_b,
        # 'bias corrected Lambda minimum MSE 2 c': estimated_mean_time_bias_corrected_Lambda_minimum_MSE_2_c

        'E minimum MSE': estimated_mean_time_E_minimum_MSE,
        'avg Lambda, E minimum MSE': avg_Lambda_E_minimum_MSE,

        'minimum MSE direct': estimated_mean_time_minimum_MSE_direct,

        'MLE': estimated_mean_time_MLE
    }


# VERSIONS_INFORMED = ['entrances', 'exits', 'entrances and exits weighted average', 'entrances and exits unweighted average', 'union']
def estimate_mean_duration_versions_informed(entrance_times, exit_times, t1, t2):
    assert len(entrance_times) == len(exit_times)

    durations = exit_times - entrance_times
    entrance_inclusions = (t1 < entrance_times) & (entrance_times <= t2)
    exit_inclusions = (t1 < exit_times) & (exit_times <= t2)
    
    durations_entrances = durations[entrance_inclusions]
    durations_exits = durations[exit_inclusions]
    durations_union = durations[entrance_inclusions | exit_inclusions]
    durations_intersection = durations[entrance_inclusions & exit_inclusions]
    n_entrances = len(durations_entrances)
    n_exits = len(durations_exits)
    # print(len(durations_entrances), len(durations_exits), len(durations_union))

    # M = np.vstack((entrance_inclusions, exit_inclusions, entrance_inclusions | exit_inclusions))
    # plt.matshow(M, aspect='auto')
    # plt.show()
    # exit()

    # plt.hist(durations_entrances, bins='auto', label='durations_entrances')
    # plt.legend()
    # plt.figure()
    # plt.hist(durations_union, bins='auto', label='durations_union')
    # plt.show()
    # exit()
    
    v_entrances = np.mean(durations_entrances)
    v_exits = np.mean(durations_exits)
    v_union = np.mean(durations_union)

    v_union_2 = (np.sum(durations_entrances) + np.sum(durations_exits) - np.sum(durations_intersection)) / (n_entrances + n_exits - len(durations_intersection))
    assert np.isclose(v_union_2, v_union)

    v_weighted_avg = (np.sum(durations_entrances) + np.sum(durations_exits)) / (n_entrances + n_exits)
    # alpha = n_entrances / (n_entrances + n_exits)
    # v_weighted_avg_2 = alpha * v_entrances + (1-alpha) * v_exits
    # assert np.isclose(v_weighted_avg, v_weighted_avg_2)
    v_unweighted_avg = (v_entrances + v_exits) / 2

    return {
        'entrances': v_entrances,
        'exits': v_exits,
        'entrances and exits weighted average': v_weighted_avg,
        'entrances and exits unweighted average': v_unweighted_avg,
        'union': v_union
    }


entrance_rate_constant = 15.0

# Set up entrance rate
###### Sinusoidal entrance rate
a = 0.7 * entrance_rate_constant
period = 20.0
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


##### Constant Entry Rate
# def rate(t):
#     if t < 0:
#         return 0
#     return entrance_rate_constant
# def cum_rate(t):
#     if t < 0:
#         return 0
#     return entrance_rate_constant * t
# def cum_rate_inverse(y):
#     assert y >= 0
#     return y / entrance_rate_constant


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


t_sample_start = 95.0
n_sample_ends = 20
max_sample_end = 150.0
t_sample_ends = np.linspace(102.0, max_sample_end, num=n_sample_ends)
n_simulations = 1000

biases_wrt_mu = {v: np.empty(n_sample_ends) for v in VERSIONS}
rmses_wrt_mu = {v: np.empty(n_sample_ends) for v in VERSIONS}
rmses_wrt_mulambda = {v: np.empty(n_sample_ends) for v in VERSIONS}
rmses_wrt_mue = {v: np.empty(n_sample_ends) for v in VERSIONS}


# The i'th element of `estimations` corresponds to estimations
# for t_sample_end being t_sample_ends[i].
estimations = [
    {v: np.empty(n_simulations) for v in VERSIONS}
    for _ in range(n_sample_ends)]

for simulation_num in trange(n_simulations):
    entrance_times, exit_times = iot.sample_realization(max_sample_end)

    n_entrances_before = len(entrance_times[entrance_times <= t_sample_start])
    n_exits_before = len(exit_times[exit_times <= t_sample_start])
    n_people_at_start = n_entrances_before - n_exits_before

    for i, t_sample_end in enumerate(t_sample_ends):
        sample_entrance_times = entrance_times[(t_sample_start < entrance_times) & (entrance_times <= t_sample_end)]
        sample_exit_times = exit_times[(t_sample_start < exit_times) & (exit_times <= t_sample_end)]

        versions_uninformed = estimate_mean_duration_versions_uninformed(n_people_at_start, sample_entrance_times, sample_exit_times, t_sample_start, t_sample_end)
        for v in VERSIONS_UNINFORMED:
            estimations[i][v][simulation_num] = versions_uninformed[v]
        
        versions_informed = estimate_mean_duration_versions_informed(entrance_times, exit_times, t_sample_start, t_sample_end)
        for v in VERSIONS_INFORMED:
            estimations[i][v][simulation_num] = versions_informed[v]

for i, t_sample_end in enumerate(t_sample_ends):
    est_means = estimations[i]
    for name in VERSIONS:
        samples = est_means[name]
        avg = np.mean(samples)
        bias_wrt_mu = avg - iot._mean_duration
        rmse_wrt_mu = np.sqrt(np.mean((samples - iot._mean_duration)**2))
        rmse_wrt_mulambda = np.sqrt(np.mean((samples - est_means['entrances'])**2))
        rmse_wrt_mue = np.sqrt(np.mean((samples - est_means['exits'])**2))
        biases_wrt_mu[name][i] = (bias_wrt_mu)
        rmses_wrt_mu[name][i] = (rmse_wrt_mu)
        rmses_wrt_mulambda[name][i] = (rmse_wrt_mulambda)
        rmses_wrt_mue[name][i] = rmse_wrt_mue


# https://matplotlib.org/2.0.2/examples/color/color_cycle_demo.html
# https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rc('axes', prop_cycle=(cycler('color', default_cycle * 2) +
                           cycler('linestyle', ['-']*10 + ['--']*10)))


# Compare uninformed estimators
plt.figure()
for v in VERSIONS_UNINFORMED + ['entrances']:
    plt.plot(t_sample_ends, rmses_wrt_mulambda[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu^\lambda$')
plt.title('RMSE of uninformed estimators w.r.t. $\mu^\lambda$')
plt.legend()

plt.figure()
for v in VERSIONS_UNINFORMED + ['exits']:
    plt.plot(t_sample_ends, rmses_wrt_mue[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu^e$')
plt.title('RMSE of uninformed estimators w.r.t. $\mu^e$')
plt.legend()

plt.figure()
for v in VERSIONS_UNINFORMED + ['entrances', 'entrances and exits unweighted average']:
    plt.plot(t_sample_ends, rmses_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu$')
plt.title('RMSE of uninformed estimators w.r.t. $\mu$')
plt.legend()

plt.figure()
for v in VERSIONS_UNINFORMED + ['entrances', 'entrances and exits unweighted average']:
    plt.plot(t_sample_ends, biases_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('bias w.r.t. $\mu$')
plt.title('bias of uninformed estimators w.r.t. $\mu$')
plt.legend()

# Compare informed estimators
plt.figure()
for v in VERSIONS_INFORMED:
    plt.plot(t_sample_ends, rmses_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('RMSE w.r.t. $\mu$')
plt.title('RMSE of informed estimators w.r.t. $\mu$')
plt.legend()

plt.figure()
for v in VERSIONS_INFORMED:
    plt.plot(t_sample_ends, biases_wrt_mu[v], label=v)
plt.xlabel('sample end time')
plt.ylabel('bias w.r.t. $\mu$')
plt.title('bias of informed estimators w.r.t. $\mu$')
plt.legend()

plt.figure()
for v in VERSIONS_INFORMED:
    mse = rmses_wrt_mu[v]**2
    bias = biases_wrt_mu[v]
    variance = mse - bias**2
    standard_error = np.sqrt(variance)
    plt.plot(t_sample_ends, standard_error, label=v)
plt.xlabel('sample end time')
plt.ylabel('standard error')
plt.title('standard error of informed estimators')
plt.legend()


plt.show()
