import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
import matplotlib.pyplot as plt
from tictoc import *
import matplotlib.pyplot as plt
from simulation import plot_pace_vs_intersection_direction
import cProfile


def get_indepenent_velocities_and_distances_func(vfunc, dfunc):
    return lambda n: np.stack((vfunc(n), dfunc(n))).T


def get_velocity_func_from_speed_func(speed_func, clockwise_prob=0.5):
    def f(n):
        velocities = speed_func(n)
        velocities[np.random.random(n) < clockwise_prob] *= -1
        return velocities

    return f


def get_choice_speed_func(inverse_speeds, probs=None):
    speeds = 1 / np.array(inverse_speeds)
    return lambda n: np.random.choice(speeds, size=n, p=probs)


def get_simple_sim(distance, dt, entrance_rate, mile_time, n_times_around,
                   same_direction_prop):
    rate_func = lambda t: entrance_rate
    rate_integral_func = lambda t: entrance_rate * t

    def get_vdist(n):
        ret = np.ones((n, 2)) * [1 / mile_time, n_times_around * distance]
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


def get_simple_sim_multiple_mile_time(distance, dt, entrance_rate, mile_times,
                                      mile_times_probs, n_times_around,
                                      same_direction_prop):
    rate_func = lambda t: entrance_rate
    rate_integral_func = lambda t: entrance_rate * t

    velocity_func = get_velocity_func_from_speed_func(
        get_choice_speed_func(mile_times, mile_times_probs),
        1 - same_direction_prop)

    get_vdist = get_indepenent_velocities_and_distances_func(
        velocity_func, lambda n: np.ones(n) * n_times_around * distance)

    # return FreshPondSim(distance,
    #                     0,
    #                     dt,
    #                     [0], [1],
    #                     get_vdist,
    #                     rate_func,
    #                     entrance_rate_integral=rate_integral_func,
    #                     interpolate_rate=False,
    #                     interpolate_rate_integral=False,
    #                     snap_exit=False)

    return FreshPondSim(distance,
                        0,
                        dt,
                        [0, 0.3*distance, 0.5*distance, 0.7*distance], [1, 1, 1, 1],
                        get_vdist,
                        rate_func,
                        entrance_rate_integral=rate_integral_func,
                        interpolate_rate=False,
                        interpolate_rate_integral=False,
                        snap_exit=False)


def plot_pace_vs_intersection_direction_simple(
        sim_dist, entrance_rate, mile_time, distance_prop,
        observer_distance_prop, same_direction_prop, max_observer_mile_time,
        mile_time_sample_interval, n_reps):
    # all the observers go this distance
    observer_distance = observer_distance_prop * sim_dist
    # the amount of time it takes the slowest observer to go the distance
    max_observer_dt = observer_distance * max_observer_mile_time

    # automatically chosen duration for the simulation
    # chosen by multiplying max_observer_dt by some factor
    # needs some time at the beginning for the simulation to reach steady state
    dt = max_observer_dt * 5

    sim = get_simple_sim(sim_dist, dt, entrance_rate, mile_time, distance_prop,
                         same_direction_prop)

    # mile times of observers
    mile_times = np.arange(0.01, max_observer_mile_time,
                           mile_time_sample_interval)
    # start time of the observers
    start_time = dt - max_observer_dt

    plot_pace_vs_intersection_direction(sim,
                                        start_time=start_time,
                                        start_pos=0,
                                        observer_distance=observer_distance,
                                        mile_times=mile_times,
                                        n_reps=n_reps)

    # cProfile.runctx('''plot_pace_vs_intersection_direction(sim,start_time=start_time,start_pos=0,observer_distance=observer_distance,mile_times=mile_times,n_reps=n_reps)''', globals(), locals(), sort='tottime')


def plot_pace_vs_intersection_direction_simple_multiple_mile_time(
        sim_dist, entrance_rate, mile_times_choices, mile_times_probs,
        distance_prop, observer_distance_prop, same_direction_prop,
        max_observer_mile_time, mile_time_sample_interval, n_reps):
    # all the observers go this distance
    observer_distance = observer_distance_prop * sim_dist
    # the amount of time it takes the slowest observer to go the distance
    max_observer_dt = observer_distance * max_observer_mile_time

    # automatically chosen duration for the simulation
    # chosen by multiplying max_observer_dt by some factor
    # needs some time at the beginning for the simulation to reach steady state
    dt = max_observer_dt * 5

    sim = get_simple_sim_multiple_mile_time(sim_dist, dt, entrance_rate,
                                            mile_times_choices,
                                            mile_times_probs, distance_prop,
                                            same_direction_prop)

    # mile times of observers
    mile_times = np.arange(0.01, max_observer_mile_time,
                           mile_time_sample_interval)
    # start time of the observers
    start_time = dt - max_observer_dt

    people_counts = plot_pace_vs_intersection_direction(sim,
                                        start_time=start_time,
                                        start_pos=0,
                                        observer_distance=observer_distance,
                                        mile_times=mile_times,
                                        n_reps=n_reps)

    # cProfile.runctx('''plot_pace_vs_intersection_direction(sim,
    #                                     start_time=start_time,
    #                                     start_pos=0,
    #                                     observer_distance=observer_distance,
    #                                     mile_times=mile_times,
    #                                     n_reps=n_reps)''', globals(), locals(), sort='tottime')

    return mile_times, people_counts, sim, start_time


def predict_true_averages(sim_dist, entrance_rate, mile_times_choices, mile_times_probs, distance_prop, observer_distance_prop, same_direction_prop, mile_times):
    tmp = entrance_rate * distance_prop * observer_distance_prop * sim_dist
    w = np.array(mile_times_choices).reshape(-1, 1)
    P = np.array(mile_times_probs).reshape(-1, 1)
    nsametot = same_direction_prop * tmp * (abs(w - mile_times) * P).sum(axis=0) / np.sum(P)

    mean_w = (w * P).sum() / P.sum()

    ndifftot = (1 - same_direction_prop) * tmp * (mile_times + mean_w)
    return nsametot, ndifftot


def test_theory():
    sim_dist = 2.46
    entrance_rate = 2.2
    mile_times_choices = [5,     10,     11,     9,   20,  23,  15,  18,  19, 21]
    mile_times_probs =   [0.05, 0.05, 0.025, 0.025, 0.25, 0.2, 0.1, 0.1, 0.1, 0.1]
    distance_prop = 0.6
    observer_distance_prop = 1.0
    same_direction_prop = 0.5
    max_observer_mile_time = 40
    mile_time_sample_interval = 4.0

    mile_times, people_counts, sim, start_time = plot_pace_vs_intersection_direction_simple_multiple_mile_time(
        sim_dist=sim_dist,
        entrance_rate=entrance_rate,
        mile_times_choices=mile_times_choices,
        mile_times_probs=mile_times_probs,
        distance_prop=distance_prop,
        observer_distance_prop=observer_distance_prop,
        same_direction_prop=same_direction_prop,
        max_observer_mile_time=max_observer_mile_time,
        mile_time_sample_interval=mile_time_sample_interval,
        n_reps=1)

    # mile_times = np.arange(0.01, max_observer_mile_time,
                           # mile_time_sample_interval)

    nsame, ndiff = predict_true_averages(sim_dist, entrance_rate, mile_times_choices, mile_times_probs, distance_prop, observer_distance_prop, same_direction_prop, mile_times)
    plt.plot(mile_times, nsame, '-', mile_times, ndiff, '-')
    plt.legend(['num same', 'num diff', 'num same pred', 'num diff pred'], loc='best')



    people_counts = list(people_counts)
    for i in range(400):
        people_counts.append(sim.n_people(start_time))
        sim.refresh_pedestrians()

    mean_count = np.mean(people_counts)
    std_count = np.std(people_counts, ddof=1)
    count_std_err = std_count / np.sqrt(len(people_counts))
    expected_mean_count = entrance_rate * np.dot(mile_times_choices, mile_times_probs) * distance_prop * sim_dist
    print("People count mean:", mean_count)
    print("People count std:", std_count)
    print("People count mean standard error:", count_std_err)
    print("Expected mean count:", expected_mean_count)






def main():
    # tic()
    # plot_pace_vs_intersection_direction_simple(sim_dist=1,
    #                                            entrance_rate=1,
    #                                            mile_time=10,
    #                                            distance_prop=3.0,
    #                                            observer_distance_prop=2,
    #                                            same_direction_prop=0.25,
    #                                            max_observer_mile_time=20,
    #                                            mile_time_sample_interval=0.25,
    #                                            n_reps=500)

    # plot_pace_vs_intersection_direction_simple_multiple_mile_time(
    #     sim_dist=2,
    #     entrance_rate=3,
    #     mile_times_choices=[10, 20],
    #     mile_times_probs=[0.2, 0.8],
    #     distance_prop=2.0,
    #     observer_distance_prop=2.4,
    #     same_direction_prop=0.25,
    #     max_observer_mile_time=40,
    #     mile_time_sample_interval=0.25,
    #     n_reps=10)
    # toc()

    test_theory()

    plt.show()


if __name__ == '__main__':
    main()
