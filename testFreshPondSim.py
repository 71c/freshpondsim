from freshpondsim import FreshPondSim, FreshPondPedestrian
from simulation import *


def assert_equals(a, b):
    if a != b:
        raise ValueError(f"{a} is not equal to {b}")


def slow_n_people(sim, t):
    n = 0
    for q in sim.pedestrians:
        if q.is_in_range(t):
            n += 1
    return n


if __name__ == '__main__':
    sim = FreshPondSim(DISTANCE, 0, DAY_LENGTH, ENTRANCES, ENTRANCE_WEIGHTS, day_rate_func, rand_velocities_and_distances)

    spacing = 0.5

    for t in np.arange(sim.start_time - 5 * spacing, sim.stop_time + 5 * spacing, spacing):
        assert slow_n_people(sim, t) == sim.n_people(t)

    # print()

    # for p in sim.pedestrians:
    #     # print(p.start_time, p.end_time)

    #     # print(slow_n_people(sim, p.start_time))

    #     assert slow_n_people(sim, p.start_time) == sim.n_people(p.start_time)
    #     # assert slow_n_people(sim, p.end_time) == sim.n_people(p.end_time)
    #     assert_equals(slow_n_people(sim, p.end_time), sim.n_people(p.end_time))

    p = FreshPondPedestrian(1, 0, 0, 0, time_delta=10)
    print(p)
