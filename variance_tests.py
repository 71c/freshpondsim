from inout_theory import InOutTheory, InOutSimulation
import scipy.stats
import numpy as np
from tqdm import tqdm
import cProfile
from tictoc import *
import matplotlib.pyplot as plt
import bisect
from generalized_binomial_utils import random_bernoulli_vectors_conditional_on_sum, get_bernoulli_probs_conditional_on_sum



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

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    for t1i, S in tqdm(zip(sim.entrance_times[:gt_t0_index], probs_before)):
        residual_time_before = t0 - t1i
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
    sim.refresh()
    nt = sim.n_people(t0) + 2

    data = np.empty((n_simulations, len(sim.entrance_times)))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations()
        while sim.n_people(t0) != nt:
            sim.refresh_durations()
        data[i] = sim.get_people_inclusions(t0).astype(float)

    T = sim.inout_theory.T

    est_conditional_probs = data.mean(axis=0)
    true_probs = T.sf(t0 - sim.entrance_times) * (sim.entrance_times <= t0).astype(float)

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])
    conditional_probs_before_matrix = get_bernoulli_probs_conditional_on_sum(probs_before, use_fft=True)
    n_zeros_per_row = len(sim.entrance_times[gt_t0_index:])
    conditional_probs = np.concatenate((conditional_probs_before_matrix[nt - 1], np.zeros(n_zeros_per_row)))

    print('Min difference:', np.min(conditional_probs - est_conditional_probs))
    print('Max difference:', np.max(conditional_probs - est_conditional_probs))

    N = true_probs.shape[0]
    x = np.arange(N)
    plt.plot(x, true_probs, 'bo', ms=1, label='unconditional probabilities')
    # plt.plot(x, true_probs * nt / np.sum(true_probs), 'go', ms=1, label='approximate conditional probabilities')
    plt.plot(x, est_conditional_probs, 'ro', ms=1, label='sample conditional probabilities')
    plt.plot(x, conditional_probs, 'go', ms=1, label='conditional probabilities')
    plt.legend()
    plt.show()


def explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people_fast(sim, t0, n_simulations):
    sim.refresh()
    nt = sim.n_people(t0)

    T = sim.inout_theory.T

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    inclusions_arr_before = random_bernoulli_vectors_conditional_on_sum(probs_before, s=nt, n=n_simulations)
    n_zeros_per_row = len(sim.entrance_times[gt_t0_index:])
    zeros = np.zeros((n_simulations, n_zeros_per_row), dtype=int)
    inclusions_arr = np.hstack((inclusions_arr_before, zeros))

    # data = np.empty((n_simulations, len(sim.entrance_times)))
    # for i in tqdm(range(n_simulations)):
    #     sim.refresh_durations_conditional_on_inclusion(t0, inclusions=inclusions_arr[i])
    #     assert sim.n_people(t0) == nt
    #     data[i] = sim.get_people_inclusions(t0).astype(float)
    # assert np.all(data == inclusions_arr)
    # est_probs = data.mean(axis=0)

    est_conditional_probs = inclusions_arr.mean(axis=0)    
    true_probs = T.sf(t0 - sim.entrance_times) * (sim.entrance_times <= t0).astype(float)

    conditional_probs_before_matrix = get_bernoulli_probs_conditional_on_sum(probs_before, use_fft=True)
    conditional_probs = np.concatenate((conditional_probs_before_matrix[nt - 1], np.zeros(n_zeros_per_row)))

    # print(list(probs_before))
    print('Min difference:', np.min(conditional_probs - est_conditional_probs))
    print('Max difference:', np.max(conditional_probs - est_conditional_probs))


    N = true_probs.shape[0]
    x = np.arange(N)
    plt.plot(x, true_probs, 'bo', ms=1, label='unconditional probabilities')
    # plt.plot(x, true_probs * nt / np.sum(true_probs), 'go', ms=1, label='approximate conditional probabilities')
    plt.plot(x, est_conditional_probs, 'ro', ms=1, label='sample conditional probabilities')
    plt.plot(x, conditional_probs, 'go', ms=1, label='conditional probabilities')
    plt.legend()
    plt.show()


def bootstrap_variance(X, g, B):
    Tboot = np.empty(B)
    n = len(X)
    for i in range(B):
        Xstar = np.random.choice(X, size=n, replace=True)
        Tboot[i] = g(Xstar)
    return np.var(Tboot, ddof=0)


def bootstrap_mean_and_se(X, g, B):
    T = g(X)
    se = np.sqrt(bootstrap_variance(X, g, B))
    return T, se


def normal_confidence_interval(x, se, alpha):
    zalphaover2 = scipy.stats.norm.ppf(1 - alpha/2)
    return (x - zalphaover2 * se, x + zalphaover2 * se)


def get_wald_pval(theta_hat, theta_0, se_hat):
    w = (theta_hat - theta_0) / se_hat
    return 2 * scipy.stats.norm.cdf(-np.abs(w))


def total_residual_time_conditional_on_entry_process_and_n_people_test(sim, t0, n_simulations):
    sim.refresh()
    nt = sim.n_people(t0) + 5

    T = sim.inout_theory.T

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    print(f'Generating inclusions conditional on ñ(t)={nt}')
    inclusions_arr_before = random_bernoulli_vectors_conditional_on_sum(probs_before, s=nt, n=n_simulations)
    n_zeros_per_row = len(sim.entrance_times[gt_t0_index:])
    zeros = np.zeros((n_simulations, n_zeros_per_row), dtype=int)
    inclusions_arr = np.hstack((inclusions_arr_before, zeros))

    print('Running simulations')
    samples = np.empty(n_simulations)
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations_conditional_on_inclusion(t0, inclusions=inclusions_arr[i])
        samples[i] = sim.total_residual_time_after(t0)

    ####### Calculate theoretical mean and variance ##########
    print('Calculating theoretical mean and variance')

    conditional_probs_before_matrix = get_bernoulli_probs_conditional_on_sum(probs_before, use_fft=True)
    conditional_probs_before = conditional_probs_before_matrix[nt - 1]

    # starting values
    theoretical_mean = -nt * t0
    theoretical_var = 0.0
    for i in tqdm(range(gt_t0_index)):
        t1i = sim.entrance_times[i]
        prob = conditional_probs_before[i]

        residual_time_before = t0 - t1i
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)

        alpha_i = t1i + conditional_mean
        theoretical_mean += alpha_i * prob
        # theoretical_var += prob * (conditional_variance + alpha_i**2 * (1 - prob))     # DEAD WRONG
        theoretical_var += conditional_variance * prob                # WRONG (but closest to right)
        # theoretical_var += conditional_variance * prob + (alpha_i - t0)**2 * prob * (1 - prob)   # WRONG

    ######## compare estimated and theoretical mean and variance #######
    sample_var, sample_var_se = bootstrap_mean_and_se(samples, np.var, 4000)
    sample_mean = np.mean(samples)
    sample_mean_se = np.sqrt(sample_var / n_simulations)

    pval_mean = get_wald_pval(sample_mean, theoretical_mean, sample_mean_se)
    pval_var = get_wald_pval(sample_var, theoretical_var, sample_var_se)

    sample_var_a, sample_var_b = normal_confidence_interval(sample_var, sample_var_se, 0.05)
    sample_mean_a, sample_mean_b = normal_confidence_interval(sample_mean, sample_mean_se, 0.05)

    # expectation is correct, variance is incorect
    print('Total residual (after) time test, conditional on entry process and n people')
    print(f'Observed mean: {sample_mean:.5f}, CI: ({sample_mean_a:.5f}, {sample_mean_b:.5f}), pval: {pval_mean:.5g} theoretical mean: {theoretical_mean:.5f}')
    print(f'Observed variance: {sample_var:.5f}, CI: ({sample_var_a:.5f}, {sample_var_b:.5f}), pval: {pval_var:.5g} theoretical variance: {theoretical_var:.5f}')


def explore_covariances_of_iverson_brackets_time_conditional_on_entry_process_and_n_people(sim, t0, n_simulations):
    sim.refresh()
    nt = sim.n_people(t0)

    T = sim.inout_theory.T

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    inclusions_arr_before = random_bernoulli_vectors_conditional_on_sum(probs_before, s=nt, n=n_simulations)
    
    covariance_matrix = np.cov(inclusions_arr_before, rowvar=False)

    theoretical_covariance_matrix = np.zeros(covariance_matrix.shape)
    N = covariance_matrix.shape[0]
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])
    conditional_probs_before_matrix = get_bernoulli_probs_conditional_on_sum(probs_before, use_fft=True)
    conditional_probs_before = conditional_probs_before_matrix[nt - 1]

    for i in tqdm(range(N)): 
        S_i = conditional_probs_before[i]
        theoretical_covariance_matrix[i, i] = S_i * (1 -  S_i)
    
    print(np.min(theoretical_covariance_matrix - covariance_matrix))
    print(np.max(theoretical_covariance_matrix - covariance_matrix))

    plt.matshow(covariance_matrix)
    plt.matshow(theoretical_covariance_matrix)
    plt.matshow(covariance_matrix - theoretical_covariance_matrix)
    plt.show()


def total_residual_time_conditional_on_entry_process_and_n_people_test_variance_decomposition(sim, t0, n_simulations):
    sim.refresh()
    T = sim.inout_theory.T
    nt = sim.n_people(t0)
    # nt = 42

    gt_t0_index = bisect.bisect(sim.entrance_times, t0)
    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])

    ######## Run simulations about the space of j_t conditional on ñ(t) ########

    print(f'Generating inclusions conditional on ñ(t)={nt}')
    inclusions_arr_before = random_bernoulli_vectors_conditional_on_sum(probs_before, s=nt, n=n_simulations, do_tqdm=True)
    n_zeros_per_row = len(sim.entrance_times[gt_t0_index:])
    zeros = np.zeros((n_simulations, n_zeros_per_row), dtype=int)
    inclusions_arr = np.hstack((inclusions_arr_before, zeros))

    print('Calculating conditional means and variances')
    conditional_means = np.empty(gt_t0_index)
    conditional_variances = np.empty(gt_t0_index)
    for i in tqdm(range(gt_t0_index)):
        t1i = sim.entrance_times[i]
        residual_time_before = t0 - t1i
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)
        conditional_means[i] = conditional_mean
        conditional_variances[i] = conditional_variance

    print('Running simulations')
    variances = np.empty(n_simulations)
    means = np.empty(n_simulations)
    means2 = np.empty(n_simulations)
    means_to_sum = np.empty((n_simulations, gt_t0_index))
    for i in tqdm(range(n_simulations)):
        sim.refresh_durations_conditional_on_inclusion(t0, inclusions=inclusions_arr[i])

        inclusions = sim.get_people_inclusions(t0)
        included_indices = np.arange(len(sim.entrance_times))[inclusions]
        
        variances[i] = np.sum(conditional_variances[included_indices])
        
        means[i] = np.sum(sim.get_entrance_times(t0)) - t0 * nt + np.sum(conditional_means[included_indices])
        
        # constants of translation do not affect variance
        means_to_sum[i] = inclusions[:gt_t0_index] * (sim.entrance_times[:gt_t0_index] + conditional_means)
        means2[i] = np.sum(sim.entrance_times[included_indices] + conditional_means[included_indices])

    ####### Calculate the theoretical expected variance and variance of expectation #######
    print('Calculating theoretical mean and variance')

    probs_before = T.sf(t0 - sim.entrance_times[:gt_t0_index])
    conditional_probs_before_matrix = get_bernoulli_probs_conditional_on_sum(probs_before, use_fft=True)
    conditional_probs_before = conditional_probs_before_matrix[nt - 1]

    # starting values
    theoretical_mean = -nt * t0
    theoretical_expected_variance = 0.0
    theoretical_variance_of_expectation = 0.0
    theoretical_covariance_matrix = np.zeros((gt_t0_index, gt_t0_index))
    for i in tqdm(range(gt_t0_index)):
        t1i = sim.entrance_times[i]
        prob = conditional_probs_before[i]

        residual_time_before = t0 - t1i
        conditional_mean = T.expect(lb=residual_time_before, conditional=True)
        conditional_variance = T.expect(
            func=lambda x: (x - conditional_mean)**2,
            lb=residual_time_before, conditional=True)

        alpha_i = t1i + conditional_mean
        theoretical_mean += alpha_i * prob
        theoretical_expected_variance += conditional_variance * prob

        theoretical_variance_of_expectation += alpha_i**2 * prob * (1 - prob)
        theoretical_covariance_matrix[i, i] = alpha_i**2 * prob * (1 - prob)

    
    covariance_matrix = np.cov(means_to_sum, rowvar=False)

    print('Theoretical covariance matrix sum:', np.sum(theoretical_covariance_matrix))
    print('Covariance matrix sum:', np.sum(covariance_matrix))

    print('Min diff covariance:', np.min(theoretical_covariance_matrix - covariance_matrix))
    print('Max diff covariance:', np.max(theoretical_covariance_matrix - covariance_matrix))

    # plt.matshow(covariance_matrix)
    # plt.matshow(theoretical_covariance_matrix)
    # plt.matshow(theoretical_covariance_matrix - covariance_matrix)
    # plt.show()

    ####### Now finally do statistics

    expected_variance, expected_variance_se = np.mean(variances), np.sqrt(np.var(variances) / n_simulations)
    variance_of_expectation, variance_of_expectation_se = bootstrap_mean_and_se(means, np.var, 1000)
    expected_expectation, expected_expectation_se = np.mean(means), np.sqrt(np.var(means) / n_simulations)
    variance_of_expectation_2, variance_of_expectation_2_se = bootstrap_mean_and_se(means2, np.var, 1000)

    pval_expected_variance = get_wald_pval(expected_variance, theoretical_expected_variance, expected_variance_se)
    pval_variance_of_expectation = get_wald_pval(variance_of_expectation, theoretical_variance_of_expectation, variance_of_expectation_se)
    pval_expectation = get_wald_pval(expected_expectation, theoretical_mean, expected_expectation_se)

    # Expected value is correct
    # Expected variance is correct
    # Variance of expectation is incorrect

    print(f'Expected mean (sample): {expected_expectation:.5f} ± {expected_expectation_se:.5f}, Expectation (theoretical): {theoretical_mean:.5f}, p-value: {pval_expectation:.5g}')
    print(f'Expected variance (sample): {expected_variance:.5f} ± {expected_variance_se:.5f}, Expected variance (theoretical): {theoretical_expected_variance:.5f}, p-value: {pval_expected_variance:.5g}')
    print(f'Variance expectation: {variance_of_expectation:.5f} ± {variance_of_expectation_se:.5f}, Variance expectation (theoretical): {theoretical_variance_of_expectation:.5f}, p-value: {pval_variance_of_expectation:.5g}')
    print(f'Variance expectation 2: {variance_of_expectation_2}')


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

# explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people(sim, t0, 4_000)
# explore_iverson_brackets_probabilities_conditional_on_entry_process_and_n_people_fast(sim, t0, 10_000)

# total_residual_time_conditional_on_entry_process_and_n_people_test(sim, t0, 3_000)

# explore_covariances_of_iverson_brackets_time_conditional_on_entry_process_and_n_people(sim, t0, 20_000)

for i in range(10):
    print(f'Run {i}:')
    total_residual_time_conditional_on_entry_process_and_n_people_test_variance_decomposition(sim, t0, 80_000)
    print()
