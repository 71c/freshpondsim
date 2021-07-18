from inout_theory import InOutTheory, InOutSimulation
import scipy.stats
import numpy as np
from tqdm import tqdm
import cProfile
from tictoc import *
import matplotlib.pyplot as plt
import bisect
from generalized_binomial_generation import random_bernoulli_vectors_conditional_on_sum



def total_residual_time_unconditional_test(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    for i in tqdm(range(n_simulations)):
        sim.refresh()
        samples[i] = sim.total_residual_time_after(t0)

    theoretical_total_residual_time_mean = iot.duration_so_far_steady_state_mean() * iot.expected_n_people(t0)

    duration_dist = sim.inout_theory.T
    steady_state_residual_time_mean_square = duration_dist.expect(lambda x: x**3) / (3 * duration_dist.mean())
    theoretical_total_residual_time_variance = iot.expected_n_people(t0) * steady_state_residual_time_mean_square

    total_residual_time_mean = np.mean(samples)
    total_residual_time_variance = np.var(samples)

    print('Total residual (after) time test')
    print(f'Observed mean: {total_residual_time_mean}, theoretical steady state mean: {theoretical_total_residual_time_mean}')
    print(f'Observed variance: {total_residual_time_variance}, theoretical steady state variance: {theoretical_total_residual_time_variance}')


def total_residual_time_conditional_on_entry_process_and_jt_test(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations_conditional_on_inclusion(t0)
        samples[i] = sim.total_residual_time_after(t0)

    sample_var = np.var(samples)
    sample_mean = np.mean(samples)

    T = sim.inout_theory.T
    
    n_people = sim.n_people(t0)
    entrance_times = sim.get_entrance_times(t0)

    # starting values
    theoretical_var = 0.0
    theoretical_mean = np.sum(entrance_times) - t0 * n_people

    for t1i in entrance_times:
        residual_time_before = t0 - t1i
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)
        theoretical_mean += conditional_mean
        theoretical_var += conditional_variance

    # CORRECT!
    print('Total residual (after) time test, conditional on entry process and jt')
    print(f'Observed mean: {sample_mean}, theoretical mean: {theoretical_mean}')
    print(f'Observed variance: {sample_var}, theoretical variance: {theoretical_var}')


def explore_numerical_covariances_conditional_on_entry_process_and_jt(sim, t0, n_simulations):
    sim.refresh()
    data = np.empty((n_simulations, sim.n_people(t0)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations_conditional_on_inclusion(t0)
        data[i] = sim.get_durations(t0)

    # covariance_matrix = np.cov(data, rowvar=False)
    # plt.matshow(covariance_matrix)
    # plt.show()

    observed_variances = np.var(data, axis=0)
    
    theoretical_variances = np.empty(observed_variances.shape)
    T = sim.inout_theory.T
    entrance_times = sim.get_entrance_times(t0)
    assert entrance_times.shape == theoretical_variances.shape
    for i, t1i in enumerate(entrance_times):
        residual_time_before = t0 - t1i
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)
        theoretical_variances[i] = conditional_variance

    print(observed_variances)
    print(theoretical_variances)


def total_residual_time_conditional_on_entry_process_test(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    sim.refresh()
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations()
        samples[i] = sim.total_residual_time_after(t0)
    
    sample_var = np.var(samples)
    sample_mean = np.mean(samples)

    T = sim.inout_theory.T
    iot = sim.inout_theory

    # starting values
    theoretical_mean = 0.0
    theoretical_var = 0.0

    for t1i in tqdm(sim.entrance_times[sim.entrance_times <= t0]):
        residual_time_before = t0 - t1i
        S = T.sf(residual_time_before)
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)

        beta_i = t1i - t0 + conditional_mean
        theoretical_var += S * (conditional_variance + beta_i**2 * (1 - S))
        theoretical_mean += beta_i * S

    # Both mean and variance correct!
    print('Total residual (after) time test, conditional on entry process')
    print(f'Observed mean: {sample_mean}, theoretical mean: {theoretical_mean}')
    print(f'Observed variance: {sample_var}, theoretical variance: {theoretical_var}')


def explore_covariances_of_iverson_brackets_time_conditional_on_entry_process(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    sim.refresh()
    # nt = sim.n_people(t0)
    data = np.empty((n_simulations, len(sim.entrance_times)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations()
        # while sim.n_people(t0) != nt:
        #     sim.refresh_durations()

        data[i] = sim.get_people_inclusions(t0).astype(float)
    
    covariance_matrix = np.cov(data, rowvar=False)
    # plt.matshow(covariance_matrix)
    # plt.show()

    T = sim.inout_theory.T

    theoretical_covariance_matrix = np.zeros(covariance_matrix.shape)
    N = covariance_matrix.shape[0]
    S_vals = T.sf(t0 - sim.entrance_times)

    for i in tqdm(range(N)): 
        t1i = sim.entrance_times[i]
        S_i = S_vals[i]
        if t1i <= t0: # technically this condition will not affect the result
            theoretical_covariance_matrix[i, i] = S_i * (1 -  S_i)
    print(np.min(theoretical_covariance_matrix - covariance_matrix))
    print(np.max(theoretical_covariance_matrix - covariance_matrix))
    plt.matshow(theoretical_covariance_matrix - covariance_matrix)
    plt.show()


# Correct
def explore_iverson_brackets_probabilities_conditional_on_entry_process(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    sim.refresh()
    data = np.empty((n_simulations, len(sim.entrance_times)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations()
        data[i] = sim.get_people_inclusions(t0).astype(float)
    
    est_probs = data.mean(axis=0)

    T = sim.inout_theory.T

    true_probs = np.zeros(est_probs.shape)
    N = true_probs.shape[0]
    true_probs = T.sf(t0 - sim.entrance_times) * (sim.entrance_times <= t0).astype(float)

    print('Min difference:', np.min(true_probs - est_probs))
    print('Max difference:', np.max(true_probs - est_probs))

    x = np.arange(N)
    plt.plot(x, true_probs, 'bo', ms=1, label='true probabilities')
    plt.plot(x, est_probs, 'ro', ms=1, label='est. probabilities')
    plt.legend()
    plt.show()


def explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    sim.refresh()
    nt = sim.n_people(t0)

    data = np.empty((n_simulations, len(sim.entrance_times)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations()
        while sim.n_people(t0) != nt:
            sim.refresh_durations()
        data[i] = sim.get_people_inclusions(t0).astype(float)
    
    est_probs = data.mean(axis=0)

    T = sim.inout_theory.T

    true_probs = np.zeros(est_probs.shape)
    N = true_probs.shape[0]
    true_probs = T.sf(t0 - sim.entrance_times) * (sim.entrance_times <= t0).astype(float)

    print('Min difference:', np.min(true_probs - est_probs))
    print('Max difference:', np.max(true_probs - est_probs))

    x = np.arange(N)
    plt.plot(x, true_probs, 'bo', ms=1, label='true probabilities')
    plt.plot(x, true_probs * nt / np.sum(true_probs), 'go', ms=1, label='true probabilities conditional approximation')
    plt.plot(x, est_probs, 'ro', ms=1, label='est. probabilities')
    plt.legend()
    plt.show()


def explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people_fast(sim, t0, n_simulations):
    samples = np.empty(n_simulations)
    sim.refresh()
    nt = sim.n_people(t0)

    T = sim.inout_theory.T

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    inclusions_arr_before = random_bernoulli_vectors_conditional_on_sum(probs_before, s=nt, n=n_simulations)
    n_zeros_per_row = len(sim.entrance_times[gt_t0_index:])
    zeros = np.zeros((n_simulations, n_zeros_per_row), dtype=int)
    inclusions_arr = np.hstack((inclusions_arr_before, zeros))

    data = np.empty((n_simulations, len(sim.entrance_times)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations_conditional_on_inclusion(t0, inclusions=inclusions_arr[i])
        assert sim.n_people(t0) == nt
        data[i] = sim.get_people_inclusions(t0).astype(float)

    assert np.all(data == inclusions_arr)

    est_probs = data.mean(axis=0)

    true_probs = np.zeros(est_probs.shape)
    N = true_probs.shape[0]
    true_probs = T.sf(t0 - sim.entrance_times) * (sim.entrance_times <= t0).astype(float)

    # print(list(probs_before))
    # print('Min difference:', np.min(true_probs - est_probs))
    # print('Max difference:', np.max(true_probs - est_probs))



    x = np.arange(N)
    plt.plot(x, true_probs, 'bo', ms=1, label='unconditional probabilities')
    # plt.plot(x, true_probs * nt / np.sum(true_probs), 'go', ms=1, label='approximate conditional probabilities')
    plt.plot(x, est_probs, 'ro', ms=1, label='sample conditional probabilities')
    plt.legend()
    plt.show()


entrance_rate_constant = 15.0

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

t0 = 10 * iot._time_constant
t_end = t0 * 2

sim = InOutSimulation(iot, t_end)

# total_residual_time_unconditional_test(sim, t0, 10_000)

# pr = cProfile.Profile()
# pr.enable()
# pr.disable()
# pr.print_stats(sort='cumulative')



# tic()
# total_residual_time_conditional_on_entry_process_and_jt_test(sim, t0, 200_000)
# toc()

# explore_numerical_covariances_conditional_on_entry_process_and_jt(sim, t0, 10_000)

# total_residual_time_conditional_on_entry_process_test(sim, t0, 5_000)

# explore_covariances_of_iverson_brackets_time_conditional_on_entry_process(sim, t0, 50_000)

# explore_iverson_brackets_probabilities_conditional_on_entry_process(sim, t0, 20_000)

# explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people(sim, t0, 1_000)
explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people_fast(sim, t0, 500)
