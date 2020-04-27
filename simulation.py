import numpy as np
from freshpondsim import FreshPondSim, FreshPondPedestrian
import cProfile
import matplotlib.pyplot as plt
from tictoc import tic, toc
from simulation_defaults import *


def show_durations_distribution():
    n = 400000

    speeds, dists = rand_velocities_and_distances(n).T

    print(max(dists / abs(speeds)) / 60)

    log = False
    plt.hist(dists / abs(speeds) / 60, bins='auto', density=True, log=log, histtype='step', label='new method')
    plt.legend()
    plt.title('histogram of duration')
    plt.xlabel('duration (hours)')
    plt.ylabel('probability density')
    plt.show()


def plot_pace_vs_distance():
    n = 20000
    speeds, dists = rand_velocities_and_distances(n).T

    plt.scatter(dists, 1/abs(speeds), s=0.5**2)
    plt.title('pace vs distance')
    plt.xlabel('distance (miles)')
    plt.ylabel('pace (minutes per mile)')
    plt.show()


def main():

    # t = time()
    sim = default_sim()
    # print(time() - t)

    # print(sim.pedestrians)

    # cProfile.runctx('FreshPondSim(DISTANCE, 0, 14400, entrances, entrance_weights, λ, rand_velocities_and_distances)', globals(), locals(), sort='cumulative')

    # print(max(sim.pedestrians, key=lambda p: p.end_time - p.start_time))

    # t = time()
    # # for _ in range(10000):
    #     # (rand_velocity())
    # u = rand_velocity(10000)
    # print(time() - t)

    # plt.hist(abs(1/u), bins='auto')
    # plt.show()

    # durations = [np.log(p.end_time - p.start_time) for p in sim.pedestrians]
    # plt.hist(durations, bins='auto')
    # plt.show()

    # distances = [p.travel_distance for p in sim.pedestrians]
    # plt.hist(distances, bins='auto')
    # plt.show()

    # print(len(sim.pedestrians))

    # props = [rand_distance_prop() for _ in range(50000)]
    # plt.hist(props, bins='auto')
    # plt.show()

    start_time = 700
    mile_times = np.linspace(1, 30, 960)
    saws = []
    tic()
    for mile_time in mile_times:
        p = FreshPondPedestrian(DISTANCE, 0, DISTANCE, start_time, velocity=1/mile_time)
        end_time = p.end_time
        n_saw = sim.n_unique_people_saw(p)
        n_at_beginning = sim.n_people(start_time)
        n_at_middle = sim.n_people((start_time + end_time) / 2)
        n_at_end = sim.n_people(end_time)
        # print(f"Start time: {start_time}")
        # print(f"End time: {end_time}")
        # print(f"Pace: {mile_time} min/mi")
        # print(f"Saw: {n_saw}")
        # print(f"n at beginning: {n_at_beginning}")
        # print(f"n at middle: {n_at_middle}")
        # print(f"n at end: {n_at_end}")
        # print()

        saws.append(n_saw)
    toc()

    # plt.plot(mile_times, saws)
    # plt.show()



    def test():
        start_time = 700
        mile_times = np.linspace(1, 30, 960)
        saws = []
        for mile_time in mile_times:
            p = FreshPondPedestrian(DISTANCE, 0, DISTANCE, start_time, velocity=1/mile_time)
            end_time = p.end_time
            n_saw = sim.n_unique_people_saw(p)
            n_at_beginning = sim.n_people(start_time)
            n_at_middle = sim.n_people((start_time + end_time) / 2)
            n_at_end = sim.n_people(end_time)

            saws.append(n_saw)

    # cProfile.runctx('test()', globals(), locals(), sort='cumulative')

    # print(sim.n_people(-1000))


if __name__ == '__main__':
    main()
