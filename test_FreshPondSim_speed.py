import numpy as np
from freshpondsim import FreshPondSim
import cProfile
from tictoc import *
from simulation_defaults import *
from utils import get_random_velocities_and_distances_func
import scipy.stats
from scipy.special import gamma
import scipy.integrate


t0 = 0
t_end = 90

entrance_rate_constant = 1400

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



########### Set up time distribution

# Using Weibull distribution. I would like a generalization of the exponential
# distribution. Weibull and Gamma are both suitable for this.
mean_time = 2.0 # expected value of time spent
k = 4 # shape parameter of weibull distribution
scale = mean_time / gamma(1 + 1/k) # scale parameter of Weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)


rand_veloc_dist_func = get_random_velocities_and_distances_func(
                                    duration_dist.rvs, 1)


# pr = cProfile.Profile()
# pr.enable()

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

# pr.disable()
# pr.print_stats(sort='cumulative')


pr = cProfile.Profile()
pr.enable()

print(sim.n_people(10))

pr.disable()
pr.print_stats(sort='cumulative')
