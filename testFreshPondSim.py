from freshpondsim import FreshPondSim, FreshPondPedestrian
from simulation_defaults import *
import math


def assert_equals(a, b):
    if a != b:
        raise ValueError(f"{a} is not equal to {b}")


def slow_n_people(sim, t):
    n = 0
    for q in sim.pedestrians:
        if q.is_in_range(t):
            n += 1
    return n


def test_n_people():
    sim = FreshPondSim(DISTANCE, 0, DAY_LENGTH, ENTRANCES, ENTRANCE_WEIGHTS, day_rate_func, rand_velocities_and_distances)

    spacing = 0.5

    for t in np.arange(sim.start_time - 5 * spacing, sim.end_time + 5 * spacing, spacing):
        assert_equals(sim.n_people(t), slow_n_people(sim, t))

    for p in sim.pedestrians:
        assert_equals(sim.n_people(p.start_time), slow_n_people(sim, p.start_time))
        assert_equals(sim.n_people(p.end_time), slow_n_people(sim, p.end_time))

    sim.clear_pedestrians()
    sim.add_pedestrian(FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60))
    assert_equals(sim.n_people(-1), 0)
    assert_equals(sim.n_people(0), 1)
    assert_equals(sim.n_people(30), 1)
    assert_equals(sim.n_people(60), 0)
    assert_equals(sim.n_people(70), 0)
    sim.add_pedestrian(FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60))
    assert_equals(sim.n_people(-1), 0)
    assert_equals(sim.n_people(0), 2)
    assert_equals(sim.n_people(30), 2)
    assert_equals(sim.n_people(60), 0)
    assert_equals(sim.n_people(70), 0)


def test_add_pedestrians():
    sim = FreshPondSim(DISTANCE, 0, DAY_LENGTH, ENTRANCES, ENTRANCE_WEIGHTS, day_rate_func, rand_velocities_and_distances)

    sim.clear_pedestrians()
    sim.add_pedestrians([])
    assert_equals(sim.num_pedestrians(), 0)

    def pedestrians_generator():
        yield FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60)
        yield FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=4, time_delta=60)

    sim.add_pedestrians(pedestrians_generator())

    assert_equals(sim.num_pedestrians(), 2)



if __name__ == '__main__':
    test_n_people()
    test_add_pedestrians()


    # p = FreshPondPedestrian(1, 0, 0, 0, time_delta=10)
    # print(p)

    # p = FreshPondPedestrian(DISTANCE, 0, DISTANCE, 0, time_delta=0)

    


    # p = FreshPondPedestrian(distance_around=DISTANCE, start_pos=0, travel_distance=0, start_time=0, time_delta=math.inf)
