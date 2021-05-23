import numpy as np
from freshpondsim import FreshPondSim
import cProfile
import matplotlib.pyplot as plt
from tictoc import *
from simulation_defaults import *
from utils import get_random_velocities_and_distances_func
import scipy.stats
from scipy.special import gamma
import scipy.integrate


t0 = 0
t_end = 90

entrance_rate_constant = 2000

###### Constant Entry Rate
def entrance_rate(t):
    if t < t0:
        return 0
    return entrance_rate_constant
def entrance_rate_integral(t):
    if t < t0:
        return 0
    return entrance_rate_constant * t


###### Sinusoidal Varying Entry Rate
# a = 0.9 * entrance_rate_constant
# period = 20
# freq = 1/period
# omega = 2*np.pi * freq
# def entrance_rate(t):
#     if t < t0:
#         return 0
#     return entrance_rate_constant + a * np.cos(omega * t)
# def entrance_rate_integral(t):
#     if t < t0:
#         return 0
#     return entrance_rate_constant * t + a / omega * np.sin(omega * t)


def integrate(func, a, b):
    y, abserr = scipy.integrate.quad(func, a, b)
    return y


########### Set up time distribution

# Using Weibull distribution. I would like a generalization of the exponential
# distribution. Weibull and Gamma are both suitable for this.
mean_time = 4.5 # expected value of time spent
k = 1 # shape parameter of weibull distribution
scale = mean_time / gamma(1 + 1/k) # scale parameter of Weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)


rand_veloc_dist_func = get_random_velocities_and_distances_func(
                                    duration_dist.rvs, 1)


pr = cProfile.Profile()
pr.enable()

sim = FreshPondSim(distance=2.5,
                   start_time=t0,
                   end_time=t_end,
                   entrances=[0],
                   entrance_weights=[1],
                   rand_velocities_and_distances_func=rand_veloc_dist_func,
                   entrance_rate=entrance_rate,
                   entrance_rate_integral=entrance_rate_integral,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)

pr.disable()
pr.print_stats(sort='cumulative')


def plot_empirical_cdf(a, **kwargs):
    # https://stackoverflow.com/a/11692365/9911203
    
    x0 = np.sort(a)
    y0 = np.arange(1,len(x0)+1)/float(len(x0))

    xrange = x0[-1] - x0[0]

    x = np.concatenate((
        np.array([x0[0] - xrange * 10]),
        x0,
        np.array([x0[-1] + xrange * 10])
    ))
    y = np.concatenate((
        np.array([0]),
        y0,
        np.array([1])
    ))

    plt.step(x, y, where='post', **kwargs)
    plt.xlim(x0[0] - xrange * 0.1, x0[-1] + xrange * 0.1)
    plt.ylim(-0.2, 1.2)


def running_average(a):
    """Returns the running average of a list"""
    return np.cumsum(a) / np.arange(1, len(a) + 1)


def plot_estimated_time_cdf(times, **kwargs):
    x0 = np.sort(times)
    tprime_cdf = np.arange(1,len(x0)+1)/float(len(x0))

    running_average_inverse_times = running_average(1 / x0)

    t_cdf = running_average_inverse_times * tprime_cdf / np.mean(1/x0)
    
    xrange = x0[-1] - x0[0]

    x = np.concatenate((
        np.array([x0[0] - xrange * 10]),
        x0,
        np.array([x0[-1] + xrange * 10])
    ))
    y = np.concatenate((
        np.array([0]),
        t_cdf,
        np.array([1])
    ))

    plt.step(x, y, where='post', **kwargs)
    plt.xlim(x0[0] - xrange * 0.1, x0[-1] + xrange * 0.1)
    plt.ylim(-0.2, 1.2)



def theoretical_time_cdf_function_at_time(S, t):
    tmp = integrate(S, 0, t)
    def F(tau):
        bound = min(t, tau)
        return (integrate(S, 0, bound) - bound * S(tau)) / tmp
    return np.vectorize(F)


def theoretical_time_cdf_function_limit(S):
    expectation = integrate(S, 0, np.inf) # expectation of T
    def F(tau):
        return (integrate(S, 0, tau) - tau * S(tau)) / expectation
    return np.vectorize(F)



t = 60

times = np.array([p.end_time - p.start_time for p in sim.get_pedestrians_at_time(t)])

# times_est = 2 * np.array([t - p.start_time for p in sim.get_pedestrians_at_time(t)])
# plt.hist(times_est, bins='auto', label='est times hist')
# plt.hist(times, bins='auto', label='true times hist')
# # plt.hist(times_est/2 / times, bins='auto', label='lookee')
# plt.legend()
# plt.figure()
# times = times_est  # really bad estimation

theoretical_cdf = theoretical_time_cdf_function_at_time(duration_dist.sf, t)
limiting_theoretical_cdf = theoretical_time_cdf_function_limit(duration_dist.sf)
x = np.linspace(duration_dist.ppf(0.001), duration_dist.ppf(0.999), 100)
plt.plot(x, duration_dist.cdf(x), label='true CDF')
plt.plot(x, theoretical_cdf(x), label='sample CDF')


plot_empirical_cdf(times, label='empirical sample CDF')

plt.plot(x, limiting_theoretical_cdf(x), label='limiting sample CDF')


plot_estimated_time_cdf(times, label='empirical true CDF')


plt.title(f'sampling at time {t}; {len(times)} people at that time')
plt.legend()


est_true_mean_time = 1 / np.mean(1 / times)
true_mean_time = duration_dist.mean()
print(f"Actual mean time: {true_mean_time:.5g}")
print(f"Estimated mean time: {est_true_mean_time:.5g}")

def get_estimated_moment(n):
    return np.mean(times**(n-1)) / np.mean(1/times)

est_second_moment = get_estimated_moment(2)
true_second_moment = duration_dist.moment(2)
print(f"Actual time 2nd moment: {true_second_moment:.5g}")
print(f"Estimated time 2nd moment: {est_second_moment:.5g}")

est_third_moment = get_estimated_moment(3)
true_third_moment = duration_dist.moment(3)
print(f"Actual time 3rd moment: {true_third_moment:.5g}")
print(f"Estimated time 3rd moment: {est_third_moment:.5g}")


plt.show()

