from freshpondsim import FreshPondSim, FreshPondPedestrian
from simulation_defaults import *
import math
from tictoc import *
import cProfile


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
    # pr = cProfile.Profile()
    # pr.enable()

    sim = default_sim()

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

    # pr.disable()
    # pr.print_stats(sort='cumulative')


def test_add_pedestrians():
    sim = default_sim()

    sim.clear_pedestrians()
    sim.add_pedestrians([])
    assert_equals(sim.num_pedestrians(), 0)

    def pedestrians_generator():
        yield FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60)
        yield FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=4, time_delta=60)

    sim.add_pedestrians(pedestrians_generator())

    assert_equals(sim.num_pedestrians(), 2)


def test_interpolation():
    tic('FreshPondSim without interpolation init')
    sim1 = FreshPondSim(DISTANCE, 0, DAY_LENGTH, ENTRANCES, ENTRANCE_WEIGHTS, default_rand_velocities_and_distances, default_day_rate_func, interpolate_rate=False, interpolate_rate_integral=False)
    toc('FreshPondSim without interpolation init')

    tic('FreshPondSim with interpolation init')
    sim2 = FreshPondSim(DISTANCE, 0, DAY_LENGTH, ENTRANCES, ENTRANCE_WEIGHTS, default_rand_velocities_and_distances, default_day_rate_func, interpolate_rate=True, interpolate_rate_integral=True, interpolate_res=2.483)
    toc('FreshPondSim with interpolation init')

    print()

    def test_non_interpolation(n):
        for i in range(n):
            tic(f'FreshPondSim without interpolation reset {i+1}')
            sim1.refresh_pedestrians()
            toc(f'FreshPondSim without interpolation reset {i+1}')
        print()

    def test_interpolation(n):
        for i in range(n):
            tic(f'FreshPondSim with interpolation reset {i+1}')
            sim2.refresh_pedestrians()
            toc(f'FreshPondSim with interpolation reset {i+1}')
        print()


    # cProfile.runctx('test_non_interpolation(40)', globals(), locals(), sort='cumulative')
    # cProfile.runctx('test_interpolation(120)', globals(), locals(), sort='tottime')

    # test_non_interpolation(100)
    test_interpolation(100)


def test_lazy_pedestrian_count():

    # pr = cProfile.Profile()
    # pr.enable()

    print("Initializing simulation")
    sim = default_sim()

    assert sim._counts_are_correct == False

    print("Getting number of people at time 1000")
    print("Number of people at time 1000:", sim.n_people(1000))

    assert sim._counts_are_correct == True

    spacing = 0.5
    for t in np.arange(sim.start_time - 5 * spacing, sim.end_time + 5 * spacing, spacing):
        assert_equals(sim.n_people(t), slow_n_people(sim, t))
        assert sim._counts_are_correct == True
    
    print("Refreshing pedestrians")
    sim.refresh_pedestrians()

    assert sim._counts_are_correct == False

    print("Clearing pedestrians")
    sim.clear_pedestrians()

    assert sim._counts_are_correct == True

    assert sim.n_people(1000) == 0

    print("Refreshing pedestrians")
    sim.refresh_pedestrians()

    assert sim._counts_are_correct == False

    print("Adding a pedestrian")
    sim.add_pedestrian(FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60))

    assert sim._counts_are_correct == False

    for t in np.arange(sim.start_time - 5 * spacing, sim.end_time + 5 * spacing, spacing):
        assert_equals(sim.n_people(t), slow_n_people(sim, t))
        assert sim._counts_are_correct == True
    
    print("Adding a pedestrian")
    sim.add_pedestrian(FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=0, time_delta=60))

    assert sim._counts_are_correct == True

    for t in np.arange(sim.start_time - 5 * spacing, sim.end_time + 5 * spacing, spacing):
        assert_equals(sim.n_people(t), slow_n_people(sim, t))
        assert sim._counts_are_correct == True
    
    print("Adding 0 pedestrians")
    sim.add_pedestrians([])

    assert sim._counts_are_correct == True

    print("Getting number of people at time 1000")
    print("Number of people at time 1000:", sim.n_people(1000))

    print("Adding 1 pedestrians")
    sim.add_pedestrians([FreshPondPedestrian(DISTANCE, start_pos=0, travel_distance=DISTANCE, start_time=10, time_delta=60)])

    assert sim._counts_are_correct == False

    print("Getting number of people at time 1000")
    print("Number of people at time 1000:", sim.n_people(1000))

    # pr.disable()
    # pr.print_stats(sort='cumulative')


if __name__ == '__main__':
    

    # test_interpolation()

    print("Running test_n_people")
    test_n_people()
    print("Test passed\n")

    print("Running test_add_pedestrians")
    test_add_pedestrians()
    print("Test passed\n")

    print("Running test_lazy_pedestrian_count")
    test_lazy_pedestrian_count()
    print("Test passed")
