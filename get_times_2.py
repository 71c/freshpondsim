import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
import cProfile
import matplotlib.pyplot as plt
from tictoc import *
from simulation_defaults import *
from utils import get_random_velocities_and_distances_func
import scipy.stats
from scipy.special import gamma
import scipy.integrate
from tqdm import tqdm
from simulation_defaults import default_day_rate_func



t0 = 0
t_end = 1440

multiplier = 20

# def entrance_rate(t):
#     if t < t0:
#         return 0
#     return multiplier * (np.sin(t) + np.sin(3*t) + 6 - 0.3 * t)

entrance_rate = default_day_rate_func

# def entrance_rate_integral(t):
#     if t < t0:
#         return 0
#     # I want it to be fast. Found by Mathematica
#     return multiplier * (4/3 + 6 * t - 3/20 * t**2 - np.cos(t) - 1/3 * np.cos(3*t))


def integrate(func, a, b, *args, **kwargs):
    y, abserr = scipy.integrate.quad(func, a, b, *args, **kwargs)
    return y


# def get_expected_n_people_integral(t):
#     return integrate(
#         lambda tau: duration_dist.sf(tau) * entrance_rate_integral(t - tau),
#         0, t - t0
#     )


def estimate_mean_time(sim, entrance_times, exit_times, t1, t2):
    n_integral = sim.n_people(t2) * (t2 - t1) + np.sum(exit_times - t1) - np.sum(entrance_times - t1)
    estimated_mean_time = n_integral / ((len(entrance_times) + len(exit_times))/2)
    return estimated_mean_time


########### Set up time distribution

# Using Weibull distribution. I would like a generalization of the exponential
# distribution. Weibull and Gamma are both suitable for this.
mean_time = 38.5222 # expected value of time spent
k = 2.43265 # shape parameter of weibull distribution
scale = mean_time / gamma(1 + 1/k) # scale parameter of Weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)


rand_veloc_dist_func = get_random_velocities_and_distances_func(
                                    duration_dist.rvs, 1)


sim = FreshPondSim(distance=2.5,
                   start_time=t0,
                   end_time=t_end,
                   entrances=[0],
                   entrance_weights=[1],
                   rand_velocities_and_distances_func=rand_veloc_dist_func,
                   entrance_rate=entrance_rate,
                #    entrance_rate_integral=entrance_rate_integral,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)


t1 = 600.0 # first time point to sample
t2 = 900.0 # second time point to sample
entrance_times, exit_times = sim.get_enter_and_exit_times_in_interval(t1, t2)
entrance_exit_times = entrance_times + exit_times
entrance_times = np.array(entrance_times)
exit_times = np.array(exit_times)
est_mean_time = estimate_mean_time(sim, entrance_times, exit_times, t1, t2)

print(len(entrance_times), len(exit_times))
print(est_mean_time)



def get_h_func(k, scale):
    def fn(t):
        return 0 if t < 0 else np.exp(-(t/scale)**k)
    return np.vectorize(fn)

def get_h_grad_k_func(k, scale):
    def fn(t):
        if t < 0:
            return 0
        c = t / scale
        return -np.exp(-c**k) * c**k * np.log(c)
    return np.vectorize(fn)


n0 = sim.n_people(t1)


khat = 2.6
alpha = 0.001
n_iter = 100
for i in range(n_iter):
    scale_param = est_mean_time / gamma(1 + 1/khat)
    h = get_h_func(khat, scale_param)
    h_grad = get_h_grad_k_func(khat, scale_param)

    J = integrate(
        lambda t: np.abs(np.sum(h(t - entrance_times)) - (sim.n_people(t) - n0)),
        t1, t2, points=entrance_exit_times, limit=3000
    )
    print(J, khat)

    grad_J = integrate(
        lambda t: np.sign(np.sum(h(t - entrance_times)) - (sim.n_people(t) - n0)) * np.sum(h_grad(t - entrance_times)),
        t1, t2, points=entrance_exit_times, limit=3000
    )
    khat -= alpha * grad_J
