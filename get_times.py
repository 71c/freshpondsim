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



t0 = 0.0
t_end = 20.0

multiplier = 4.0

def entrance_rate(t):
    if t < t0:
        return 0
    return multiplier * (np.sin(t) + np.sin(3*t) + 6 - 0.3 * t)


def entrance_rate_integral(t):
    if t < t0:
        return 0
    # I want it to be fast. Found by Mathematica
    return multiplier * (4/3 + 6 * t - 3/20 * t**2 - np.cos(t) - 1/3 * np.cos(3*t))


def integrate(func, a, b):
    y, abserr = scipy.integrate.quad(func, a, b)
    return y


def get_expected_n_people_integral(t):
    return integrate(
        lambda tau: duration_dist.sf(tau) * entrance_rate_integral(t - tau),
        0, t - t0
    )


def estimate_mean_time_v1(sim, entrance_times, exit_times, t1, t2):
    n_integral = sim.n_people(t2) * (t2 - t1) + np.sum(exit_times - t1) - np.sum(entrance_times - t1)
    estimated_mean_time = n_integral / ((len(entrance_times) + len(exit_times))/2)
    return estimated_mean_time


def estimate_mean_time_v2(sim, entrance_times, exit_times, t1, t2):
    n_integral = sim.n_people(t2) * (t2 - t1) + np.sum(exit_times - t1) - np.sum(entrance_times - t1)
    estimated_mean_time = 0.5 * (n_integral / len(entrance_times) + n_integral / len(exit_times))
    return estimated_mean_time


def estimate_mean_time_v3(sim, entrance_times, exit_times, t1, t2):
    n_integral = sim.n_people(t2) * (t2 - t1) + np.sum(exit_times - t1) - np.sum(entrance_times - t1)
    estimated_mean_time = n_integral / len(exit_times)
    return estimated_mean_time


def estimate_mean_time_v4(sim, entrance_times, exit_times, t1, t2):
    n_integral = sim.n_people(t2) * (t2 - t1) + np.sum(exit_times - t1) - np.sum(entrance_times - t1)
    estimated_mean_time = n_integral / len(entrance_times)
    return estimated_mean_time


########### Set up time distribution

# Using Weibull distribution. I would like a generalization of the exponential
# distribution. Weibull and Gamma are both suitable for this.
mean_time = 4.5 # expected value of time spent
k = 3 # shape parameter of weibull distribution
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
                   entrance_rate_integral=entrance_rate_integral,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)

# t1 = 7.0 # first time point to sample
# t2 = 14.0 # second time point to sample


n_trials = 1000
mean_times_v1 = np.empty(n_trials)
mean_times_v2 = np.empty(n_trials)
mean_times_v3 = np.empty(n_trials)
mean_times_v4 = np.empty(n_trials)
for i in tqdm(range(n_trials)):
    entrance_times, exit_times = [], []
    while len(entrance_times) < 20 or len(exit_times) < 20:
        # t1 = np.random.uniform(t0, t_end)
        t1 = t0
        t2 = np.random.uniform(t1, t_end)
        entrance_times, exit_times = sim.get_enter_and_exit_times_in_interval(t1, t2)
        
    entrance_times = np.array(entrance_times)
    exit_times = np.array(exit_times)

    mean_times_v1[i] = estimate_mean_time_v1(sim, entrance_times, exit_times, t1, t2)
    mean_times_v2[i] = estimate_mean_time_v2(sim, entrance_times, exit_times, t1, t2)
    mean_times_v3[i] = estimate_mean_time_v3(sim, entrance_times, exit_times, t1, t2)
    mean_times_v4[i] = estimate_mean_time_v4(sim, entrance_times, exit_times, t1, t2)
    sim.refresh_pedestrians()


for name, samples in [('v1', mean_times_v1), ('v2', mean_times_v2), ('v3', mean_times_v3), ('v4', mean_times_v4)]:
    avg = np.mean(samples)
    # rmse = np.sqrt(np.mean((samples - mean_time)**2))
    mae = np.mean(np.abs(samples - mean_time))
    print(f"{name}: Average: {avg:.3f}, MAE: {mae:.3f}")





# t1 = 7.0 # first time point to sample
# t2 = 14.0 # second time point to sample
# entrance_times, exit_times = sim.get_enter_and_exit_times_in_interval(t1, t2)
# est_mean_time = estimate_mean_time_v1(sim, entrance_times, exit_times, t1, t2)
