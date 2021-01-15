from freshpondsim import FreshPondSim
from utils import get_random_velocities_and_distances_func
import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm
from tictoc import *


def integrate(func, a, b):
    y, abserr = scipy.integrate.quad(func, a, b)
    return y


t0 = 0

def entrance_rate(t):
    if t < t0:
        return 0
    # return np.sin(t) + 3
    # return 3

    return np.sin(t) + np.sin(3*t) + 6 - 0.3 * t


def entrance_rate_integral(t):
    # return integrate(entrance_rate, t0, t)

    return 4/3 + 6 * t - 3/20 * t**2 - np.cos(t) - 1/3 * np.cos(3*t)

    # if t < 0:
    #     return 0
    # return -np.cos(t) + 3 * t + 1



duration_dist = scipy.stats.expon(scale=4.2)
# survival function: duration_dist.sf(x)
# random variate(s): duration_dist.rvs(size=<size>)

rand_veloc_dist_func = get_random_velocities_and_distances_func(
                                    duration_dist.rvs, 1)


def get_expected_n_people(t):
    return integrate(
        lambda tau: duration_dist.sf(tau) * entrance_rate(t - tau),
        0, t
    )


sim = FreshPondSim(distance=2.5,
                   start_time=t0,
                   end_time=10,
                   entrances=[0],
                   entrance_weights=[1],
                   rand_velocities_and_distances_func=rand_veloc_dist_func,
                   entrance_rate=entrance_rate,
                   entrance_rate_integral=entrance_rate_integral,
                   interpolate_rate=False,
                   interpolate_rate_integral=True,
                   interpolate_res=0.001,
                   snap_exit=False)

t1 = 0.4 # first time point to sample
t2 = 7.4 # second time point to sample

# gather samples
n_samples = 200000
samples = []
for sample_num in tqdm(range(n_samples)):
    n1 = sim.n_people(t1)
    n2 = sim.n_people(t2)
    samples.append(n2 - n1)
    sim.refresh_pedestrians()

# construct skellam distribution
dt = t2 - t1
ne = get_expected_n_people(t1) - integrate(
                lambda tau: duration_dist.sf(tau) * entrance_rate(t2 - tau),
                dt, t2 - t0)
nlambda = integrate(lambda tau: duration_dist.sf(tau) * entrance_rate(t2 - tau),
                    0, dt)
mu1 = nlambda
mu2 = ne
skellam_dist = scipy.stats.skellam(mu1, mu2)


fig, ax = plt.subplots(1, 1)

# d = np.diff(np.unique(samples)).min()
d = 1
left_of_first_bin = np.min(samples) - float(d)/2
right_of_last_bin = np.max(samples) + float(d)/2
ax.hist(samples, np.arange(left_of_first_bin, right_of_last_bin + d, d), density=True)

x = np.arange(np.min(samples),
              np.max(samples) + 1)
ax.plot(x, skellam_dist.pmf(x), 'bo', ms=8, label='skellam pmf')
ax.legend(loc='best', frameon=False)

plt.show()

observed_variance = np.var(samples, ddof=1)
expected_variance = skellam_dist.var()
print(f"observed variance: {observed_variance:.2f}, expected variance: {expected_variance:.2f}")
print(f"observed mean: {np.mean(samples):.2f}, expected mean: {skellam_dist.mean():.2f}")

print()
print()
