from freshpondsim import FreshPondSim
from utils import get_random_velocities_and_distances_func
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from tictoc import *


################# Set up simulation #######################

t0 = 0
t_end = 60*8
entrance_rate_constant = 1000

### Constant Entry Rate
# def entrance_rate(t):
#     if t < t0:
#         return 0
#     return entrance_rate_constant
# def entrance_rate_integral(t):
#     if t < t0:
#         return 0
#     return entrance_rate_constant * (t - t0)
# def entrance_rate_integral_inverse(y):
#     assert y >= 0
#     return y / entrance_rate_constant + t0

### Sinusoidal Varying Entry Rate
a = 0.9 * entrance_rate_constant
period = 80
freq = 1/period
omega = 2*np.pi * freq
def entrance_rate(t):
    if t < t0:
        return 0
    return entrance_rate_constant + a * np.cos(omega * t)
def entrance_rate_integral(t):
    if t < t0:
        return 0
    return entrance_rate_constant * t + a / omega * np.sin(omega * t)
entrance_rate_integral_inverse = None


### Set up time distribution
scale = 42.59286560661815 # scale parameter of Weibull distribution
k = 1.5513080437971483 # shape parameter of weibull distribution
duration_dist = scipy.stats.weibull_min(k, scale=scale)
mean_duration = duration_dist.mean()


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
                   entrance_rate_integral_inverse=entrance_rate_integral_inverse,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)


########## Collect simulation data ########
print("Running simulations...")
t = 60*8-10 # t is the point in time we are looking at
n_reps = 3

times_so_far = [] # how long is it since people came of people there at time t
times_total = []  # how long people end up staying of people there at time t

times_so_far.extend(t - p.start_time for p in sim.get_pedestrians_at_time(t))
times_total.extend(p.end_time - p.start_time for p in sim.get_pedestrians_at_time(t))

for _ in tqdm(range(n_reps - 1)):
    sim.refresh_pedestrians()
    times_so_far.extend(t - p.start_time for p in sim.get_pedestrians_at_time(t))
    times_total.extend(p.end_time - p.start_time for p in sim.get_pedestrians_at_time(t))

times_so_far = np.array(times_so_far)
times_total = np.array(times_total)
average_time_so_far = np.mean(times_so_far)
average_time_total = np.mean(times_total)

print("Number of samples collected:", len(times_so_far))


######### Theory #########

def integrate(func, a, b):
    y, abserr = scipy.integrate.quad(func, a, b)
    return y


def expected_num_people(t):
    return integrate(lambda u: duration_dist.sf(u) * entrance_rate(t - u), 0, t)


def duration_so_far_density_func(t):
    n_t = expected_num_people(t)
    @np.vectorize
    def duration_so_far_density(x):
        return 0 if x < 0 else entrance_rate(t - x) * duration_dist.sf(x) / n_t
    return duration_so_far_density


def duration_density_func(t):
    n_t = expected_num_people(t)
    @np.vectorize
    def duration_density(x):
        return duration_dist.pdf(x) * (entrance_rate_integral(t) - entrance_rate_integral(t - x)) / n_t
    return duration_density


duration_so_far_density = duration_so_far_density_func(t)
duration_density = duration_density_func(t)

time_total_expectation_const_entry_rate = mean_duration + duration_dist.var() / mean_duration

time_total_expectation = integrate(lambda x: x * duration_density(x), 0, np.inf)
time_so_far_expectation = integrate(lambda x: x * duration_so_far_density(x), 0, np.inf)


print(f"Average time so far: {average_time_so_far:.5g}")
print(f"Time so far expectation: {time_so_far_expectation:.5g}")
print(f"Average time total: {average_time_total:.5g}")
print(f"Time total expectation: {time_total_expectation:.5g}")
print(f"Average time so far / average time total: {average_time_so_far/average_time_total:.5g}")
print(f"time so far expectation / time total expectation: {time_so_far_expectation/time_total_expectation:.5g}")



# max_time = duration_dist.ppf(0.999)
max_time = max(times_total)
print("Max time from simulation:", max_time, "quantile:", duration_dist.cdf(max_time))

tvals = np.linspace(0, max_time, num=100)


plt.plot(tvals, duration_dist.pdf(tvals), label='duration dist pdf')
plt.hist(times_so_far, bins='auto', density=True, label='histogram of duration so far')
plt.plot(tvals, duration_so_far_density(tvals), label='duration so far PDF')
plt.plot(tvals, duration_dist.sf(tvals)/mean_duration, label='duration so far PDF assuming constant entry rate')
plt.legend()

plt.figure()
plt.plot(tvals, duration_dist.pdf(tvals), label='duration dist pdf')
plt.hist(times_total, bins='auto', density=True, label='histogram of duration here')
plt.plot(tvals, duration_density(tvals), label='duration here PDF')
plt.plot(tvals, tvals*duration_dist.pdf(tvals)/mean_duration, label='duration here PDF assuming constant entry rate')
plt.legend()

plt.show()
