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


def get_choice_speed_func(inverse_speeds):
    speeds = 1 / np.array(inverse_speeds)
    return lambda n: np.random.choice(speeds, n)


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
                        interpolate_rate_integral=False)


def get_simple_sim_multiple_mile_time(distance, dt, entrance_rate, mile_times,
                                      n_times_around, same_direction_prop):
    rate_func = lambda t: entrance_rate
    rate_integral_func = lambda t: entrance_rate * t

    velocity_func = get_velocity_func_from_speed_func(
        get_choice_speed_func(mile_times), 1 - same_direction_prop)

    get_vdist = get_indepenent_velocities_and_distances_func(
        velocity_func, lambda n: np.ones(n) * n_times_around * distance)

    return FreshPondSim(distance,
                        0,
                        dt, [0], [1],
                        get_vdist,
                        rate_func,
                        entrance_rate_integral=rate_integral_func,
                        interpolate_rate=False,
                        interpolate_rate_integral=False)


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


def main():
    tic()
    plot_pace_vs_intersection_direction_simple(sim_dist=1,
                                               entrance_rate=1,
                                               mile_time=10,
                                               distance_prop=3.0,
                                               observer_distance_prop=2,
                                               same_direction_prop=0.25,
                                               max_observer_mile_time=20,
                                               mile_time_sample_interval=0.25,
                                               n_reps=500)
    toc()
    plt.show()


if __name__ == '__main__':
    main()
