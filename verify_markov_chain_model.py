from freshpondsim import FreshPondSim
from utils import get_random_velocities_and_distances_func
import scipy.stats
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from numpy import linalg as la

t0 = 0
t_end = 60*8

entrance_rate_constant = 1000

###### Constant Entry Rate
def entrance_rate(t):
    if t < t0:
        return 0
    return entrance_rate_constant
def entrance_rate_integral(t):
    if t < t0:
        return 0
    return entrance_rate_constant * (t - t0)
def entrance_rate_integral_inverse(y):
    assert y >= 0
    return y / entrance_rate_constant + t0

########### Set up time distribution
scale = 42.59286560661815 # scale parameter of Weibull distribution
k = 1.5513080437971483 # shape parameter of weibull distribution
mean_duration = scale * gamma(1 + 1/k)
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
                   entrance_rate_integral_inverse=entrance_rate_integral_inverse,
                   interpolate_rate=False,
                   interpolate_rate_integral=False,
                   snap_exit=False)


t = 60*8-10
n_reps = 5

times = []

times.extend(t - p.start_time for p in sim.get_pedestrians_at_time(t))

for _ in tqdm(range(n_reps - 1)):
    sim.refresh_pedestrians()
    times.extend(t - p.start_time for p in sim.get_pedestrians_at_time(t))

times = np.array(times)




def get_truncated_above_sf(cdf, maxval):
    cdf_maxval = cdf(maxval)
    return lambda t: 1 - cdf(np.minimum(t, maxval)) / cdf_maxval

def get_truncated_above_pdf(cdf, pdf, maxval):
    cdf_maxval = cdf(maxval)
    return np.vectorize(lambda t: pdf(t) / cdf_maxval if t <= maxval else 0)


dt = 0.5 # time interval in minutes

# truncate the distribution to lie below max_time
max_time = duration_dist.ppf(0.999)
n_intervals = int(np.ceil(max_time / dt))
max_time = n_intervals * dt

# survival function of truncated duration distribution
trunc_sf = get_truncated_above_sf(duration_dist.cdf, max_time)

# create transition matrix for markov chain.
# column i of the matrix is the probabilities of going to each state if
# currently in state i.
# state 0 is for being outside the res, the other states are for being at
# different levels of progress through being in the res.
# state 1 is the first state you are at when you enter the res.
P = np.zeros((n_intervals + 1, n_intervals + 1))

P[1, 0] = 1
P[0, 0] = 0

t_samples = np.linspace(0, max_time, num=n_intervals + 1)

# k goes thru all k's except 0 and n_intervals
for k in range(1, n_intervals):
    stay_prob = trunc_sf(t_samples[k]) / trunc_sf(t_samples[k-1])
    P[k + 1, k] = stay_prob
    P[0, k] = 1 - stay_prob
P[0, n_intervals] = 1

print(f'Finding eigenvalues and eigenvectors of the {n_intervals+1}x{n_intervals+1} transition matrix...')
vals, vecs = la.eig(P)
print('Done finding those eigens! :)')
eig1_index = np.argmin(abs(vals - 1))
assert np.isclose(vals[eig1_index].real, 1, rtol=0, atol=1e-10)
assert vals[eig1_index].imag == 0
assert np.all(vecs[:, eig1_index].imag == 0)

v = vecs[:, eig1_index].real
v /= v.sum()

assert np.allclose(P @ v, v)

t_probabilities = v[1:] / (1 - v[0])
t_densities = t_probabilities / dt

t_midpoints = t_samples[1:] - dt/2
trunc_pdf = get_truncated_above_pdf(duration_dist.cdf, duration_dist.pdf, max_time)
plt.plot(t_midpoints, t_densities, label='duration so far density by markov chain eigenvector approximation')
# plt.plot(t_midpoints, trunc_pdf(t_midpoints), label='truncated duration dist pdf')
plt.plot(t_midpoints, duration_dist.pdf(t_midpoints), label='duration dist pdf')
plt.hist(times, bins='auto', density=True, label='histogram of duration so far')
plt.plot(t_midpoints, duration_dist.sf(t_midpoints)/mean_duration, label='duration so far density exact')
plt.legend()
plt.show()
