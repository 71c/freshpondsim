import numpy as np

def get_random_velocities_and_distances_func(random_duration_func, distance_traveled):
    def ret(n):
        durations = random_duration_func(n)
        distances = np.ones(n) * distance_traveled
        velocities = distances / durations
        return np.array([velocities, distances]).T
    return ret
