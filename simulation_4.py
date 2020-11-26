import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
from scipy.stats import lognorm
import scipy.integrate as integrate
from tictoc import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile


def get_sim_with_mile_time_dist(distance, dt, entrance_rate, mile_time_dist,
                                n_times_around, same_direction_prop):
    rate_func = lambda t: entrance_rate
    rate_integral_func = lambda t: entrance_rate * t

    scale = 1/mile_time_mean * np.exp(sigma**2 / 2) # exp(mu)

    def get_vdist(n):
        ret = np.array([1/mile_time_dist.rvs(size=n), np.ones(n) * n_times_around * distance]).T
        ret[np.random.random(n) > same_direction_prop, 0] *= -1
        return ret

    return FreshPondSim(distance,
                        0,
                        dt, [0], [1],
                        get_vdist,
                        rate_func,
                        entrance_rate_integral=rate_integral_func,
                        interpolate_rate=False,
                        interpolate_rate_integral=False,
                        snap_exit=False)


def plot_number_of_people_over_time(sim, mile_time_dist, mean_distance, rate, n_trials):
    time_samples = np.arange(0, sim.end_time - sim.start_time, 1)
    for i in range(n_trials):
        people_counts = [sim.n_people(t) for t in time_samples]
        plt.plot(time_samples, people_counts)
        sim.refresh_pedestrians()

    a = rate * integrate.cumtrapz(mile_time_dist.sf(time_samples / mean_distance), time_samples, initial=0)
    plt.plot(time_samples, a, label='expected number of people', color='k')
    plt.plot(time_samples, a - 2 * np.sqrt(a))
    plt.plot(time_samples, a + 2 * np.sqrt(a))

    mean_duration = mile_time_dist.mean() * mean_distance

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('people')
    plt.title(f'number of people over time, {n_trials} simulations\nλ={rate}, τ={mean_duration}')
    plt.show()


def get_n_saw_samples(dist_around, mile_time_dist, entrance_rate,
                        n_times_around, same_direction_prop,
                        observer_n_times_around, observer_mile_time, t0,
                        n_samples):
    # observer_travel_distance = observer_n_times_around * dist_around
    # observer_dt = observer_mile_time * observer_travel_distance
    # sim_dt = t0 + observer_dt * n_samples
    # sim = get_sim_with_mile_time_dist(dist_around, sim_dt, entrance_rate, mile_time_dist, n_times_around, same_direction_prop)
    
    # observer_start_time = t0
    # n_sames, n_diffs = [], []

    # for i in tqdm(range(n_samples)):
    #     observer = FreshPondPedestrian(
    #         distance_around=dist_around,
    #         start_pos=0,
    #         travel_distance=observer_travel_distance,
    #         start_time=observer_start_time,
    #         velocity=1/observer_mile_time)
    #     n_same, n_diff = sim.intersection_directions(observer)
    #     n_sames.append(n_same)
    #     n_diffs.append(n_diff)
    #     observer_start_time += observer_dt

    # return sim, n_sames, n_diffs

    n_samples_per = 25
    n_runs = n_samples // n_samples_per

    observer_travel_distance = observer_n_times_around * dist_around
    observer_dt = observer_mile_time * observer_travel_distance
    sim_dt = t0 + observer_dt * n_samples_per
    sim = get_sim_with_mile_time_dist(dist_around, sim_dt, entrance_rate, mile_time_dist, n_times_around, same_direction_prop)

    n_sames, n_diffs = [], []

    for i in tqdm(range(n_runs)):
        observer_start_time = t0
        for j in range(n_samples_per):
            observer = FreshPondPedestrian(
                distance_around=dist_around,
                start_pos=0,
                travel_distance=observer_travel_distance,
                start_time=observer_start_time,
                velocity=1/observer_mile_time)
            n_same, n_diff = sim.intersection_directions(observer)
            n_sames.append(n_same)
            n_diffs.append(n_diff)
            observer_start_time += observer_dt
        sim.refresh_pedestrians()

    return sim, n_sames, n_diffs


# define some constants for the simulation
distance = 2 # distance around
dt = 400
entrance_rate = 4
n_times_around = 2
same_direction_prop = 1.0
mile_time_mean = 20

# define the mile time distribution
sigma = 0.1
scale = mile_time_mean * np.exp(-sigma**2 / 2) # exp(mu)
mile_time_dist = lognorm(sigma, loc=0, scale=scale)

# mean duration someone spends at the reservoir
mean_duration = mile_time_mean * distance * n_times_around

n_times_around_observer = 1
observer_mile_time = 10

t0 = 5 * mean_duration

tic()
sim, n_sames, n_diffs = get_n_saw_samples(distance, mile_time_dist,
    entrance_rate, n_times_around, same_direction_prop,
    n_times_around_observer, observer_mile_time, t0, n_samples=1000)
toc()
mean_n_same = np.mean(n_sames)
mean_n_diff = np.mean(n_diffs)

# cProfile.runctx('''
# sim, n_sames, n_diffs = get_n_saw_samples(distance, mile_time_dist,
#     entrance_rate, n_times_around, same_direction_prop,
#     n_times_around_observer, observer_mile_time, t0, n_samples=500)
# ''', globals(), locals(), sort='cumulative')


# tic()
# # the simulation
# sim = get_sim_with_mile_time_dist(distance, dt, entrance_rate, mile_time_dist, n_times_around, same_direction_prop)

# observer = FreshPondPedestrian(
#     distance_around=distance,
#     start_pos=0,
#     travel_distance=n_times_around_observer * distance,
#     start_time=4 * mean_duration,
#     velocity=1/observer_mile_time)

# mean_n_same, mean_n_diff = 0.0, 0.0
# n_trials = 1000
# for i in range(n_trials):
#     n_same, n_diff = sim.intersection_directions(observer)
#     mean_n_same += n_same
#     mean_n_diff += n_diff
#     sim.refresh_pedestrians()
# mean_n_same /= n_trials
# mean_n_diff /= n_trials

# toc()



print(f"mean_duration:", mean_duration)
expected_n_people = entrance_rate * mean_duration


very_wrong_expected_n_same       = same_direction_prop * expected_n_people * n_times_around_observer * mile_time_dist.expect(lambda w: np.abs(observer_mile_time / w - 1))

wrong_expected_n_same = same_direction_prop * expected_n_people * n_times_around_observer * mile_time_dist.expect(lambda w: np.abs(w - observer_mile_time)) / mile_time_mean

print('very wrong expected n same:', very_wrong_expected_n_same)
print('wrong expected n same:', wrong_expected_n_same)
print('mean n same:', mean_n_same, 'std n same:', np.std(n_sames, ddof=1))





# plot_number_of_people_over_time(sim, mile_time_dist, mean_distance=distance * n_times_around, rate=entrance_rate, n_trials=3)

# plt.hist(n_sames, bins='auto')
# plt.show()
