import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
import cProfile
import matplotlib.pyplot as plt
from tictoc import *
from simulation_defaults import *


def show_durations_distribution(n_samples=400000, log=False):
    speeds, dists = default_rand_velocities_and_distances(n_samples).T

    # durations in hours
    durations = dists / abs(speeds) / 60

    print("Maximum duration:", max(durations))

    plt.hist(durations, bins='auto', density=True, log=log, histtype='step')
    plt.legend()
    plt.title('histogram of duration')
    plt.xlabel('duration (hours)')
    plt.ylabel('probability density')
    plt.show()


def show_pace_distribution(n_samples=400000, log=False):
    speeds, dists = default_rand_velocities_and_distances(n_samples).T

    plt.hist(abs(1/speeds), bins='auto', density=True, log=log, histtype='step')
    plt.legend()
    plt.title('histogram of speed')
    plt.xlabel('duration (hours)')
    plt.ylabel('probability density')
    plt.show()


def plot_pace_vs_distance(n_samples=20000):
    speeds, dists = rand_velocities_and_distances(n_samples).T

    plt.scatter(dists, 1/abs(speeds), s=0.5**2)
    plt.title('pace vs distance')
    plt.xlabel('distance (miles)')
    plt.ylabel('pace (minutes per mile)')
    plt.show()


def get_n_unique_saws(sim, pedestrians):
    saws = np.empty(len(pedestrians))
    for i, p in enumerate(pedestrians):
        saws[i] = sim.n_unique_people_saw(p)
    return saws


def minutes_to_time_string_12hr(minutes):
    minutes = int(minutes)
    minute_part = minutes % 60
    hour_part = minutes // 60 % 24
    am_pm = 'PM' if hour_part >= 12 else 'AM'
    hour_part = hour_part % 12
    if hour_part == 0:
        hour_part = 12
    hour_part_string = str(hour_part)
    minute_part_string = str(minute_part) if minute_part > 9 else '0' + str(minute_part)
    return hour_part_string + ':' + minute_part_string + am_pm


def get_average_n_unique_saw_for_paces(sim, start_time, start_pos, distance, mile_times, n_reps):
    pedestrians = [
        FreshPondPedestrian(DISTANCE, start_pos, distance, start_time, 1/mile_time)
        for mile_time in mile_times]

    saws_avg = np.zeros(len(pedestrians))
    for _ in range(n_reps):
        saws_avg += get_n_unique_saws(sim, pedestrians)
        sim.refresh_pedestrians()
    saws_avg /= n_reps

    return saws_avg


def plot_pace_vs_n_unique_saw(sim, start_time, start_pos, distance, mile_times, n_reps=1):
    tic('get pace vs unique saw data')

    saws_avg = get_average_n_unique_saw_for_paces(
        sim, start_time, start_pos, distance, mile_times, n_reps)

    tocl()

    plt.plot(mile_times, saws_avg)
    plt.xlabel('mile time (minutes)')
    plt.ylabel('number of unique people saw')

    time_str = minutes_to_time_string_12hr(start_time)
    title_text = f'going {distance} mi starting at {time_str}'
    if n_reps > 1:
        title_text += f' (average of {n_reps} simulations)'
    plt.title(title_text)


def get_average_intersection_directions_for_paces(sim, start_time, start_pos, distance, mile_times, n_reps):
    # observer pedestrians going at different mile times
    observers = [
        FreshPondPedestrian(sim.dist_around, start_pos, distance, start_time, 1/mile_time)
        for mile_time in mile_times]

    n_same_avg = np.zeros(len(observers))
    n_diff_avg = np.zeros(len(observers))
    people_counts = np.zeros(n_reps)
    for k in range(n_reps):
        for i, p in enumerate(observers):
            # n_same, n_diff = sim.intersection_directions(p)
            n_same, n_diff = sim.intersection_directions_total(p)
            n_same_avg[i] += n_same
            n_diff_avg[i] += n_diff
        people_counts[k] = sim.n_people(start_time)
        sim.refresh_pedestrians()
    n_same_avg /= n_reps
    n_diff_avg /= n_reps

    return n_same_avg, n_diff_avg, people_counts


def plot_pace_vs_intersection_direction(sim, start_time, start_pos, observer_distance, mile_times, n_reps=1):
    n_same_avg, n_diff_avg, people_counts = get_average_intersection_directions_for_paces(
        sim, start_time, start_pos, observer_distance, mile_times, n_reps)

    plt.plot(mile_times, n_same_avg, '-', mile_times, n_diff_avg, '-')
    plt.legend(['num same', 'num diff'], loc='best')
    plt.xlabel('mile time (minutes)')
    plt.ylabel('number of unique people saw')

    time_str = minutes_to_time_string_12hr(start_time)
    title_text = f'going {observer_distance} mi starting at {time_str}'
    if n_reps > 1:
        title_text += f' (average of {n_reps} simulations)'
    plt.title(title_text)

    return people_counts



def main():
    sim = default_sim()

    # plot_pace_vs_n_unique_saw(sim, 19.5*60, 0, DISTANCE, np.arange(0.01, 30, 2), 30)

    # show_pace_distribution()

    # plot_pace_vs_intersection_direction(sim, 15*60, 0, DISTANCE, np.arange(0.01, 40, 2), 50)

    plt.show()

    # cProfile.runctx('get_average_n_unique_saw_for_paces(sim, 720, 0, DISTANCE, np.linspace(1, 30, 116), 5)', globals(), locals(), sort='cumulative')


if __name__ == '__main__':
    main()
