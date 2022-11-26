# freshpondsim
This is a set of Python scripts which are used to verify and test a bunch of work I did on mathematically modeling the movement of people around a reservoir park, which I worked on in Summer 2020 and Summer 2021. This theory includes, for example, the theory of M_t/G/∞ queues (part of queueing theory).

This code accompanies the extensive theoretical work I did, and verifies and tests a lot of the math I came up with.

The project started as modeling how people go around a reservoir, or any long circular path, throughout time.
Later, I generalized this and considered any state that objects can move in and out of. Only later, in late Summer 2021, did I discover that what I was doing was exploring the theory of what is called a M_t/G/∞ queue, and it is part of queueing theory.

## Files
The main files are:
* `freshpondsim.py`: This is the file that contains the code for the simulation program.
It contains two classes: `FreshPondPedestrian` and `FreshPondSim`.
  * A `FreshPondPedestrian` object represents a person going around the reservoir.
  * A `FreshPondSim` object represents a simulation of people going around the reservoir.
    It cointains a list  of `FreshPondPedestrian`s stored in it called `pedestirans`.
    The number of people at time `t` is accessed by `self.n_people(t)`
* `inout_theory.py`: defines two important classes:
  * The class `InOutTheory` contains code that analytically computes formulas for properties of a M_t/G/∞ queue, and also code that can generate a random realization
  * The class `InOutSimulation` is used to represent a simulation, or random realization, of a M_t/G/∞ queue. It is similar to `FreshPondSim`, but it is for a general M_t/G/∞ queue, not specifically a reservoir.
* `get_times_3_improved.py`: tests out a lot of theory for estimating the average time that someone spends in the queue given just the entrance times and exit times
* `variance_tests.py`: tests out a bunch of very complicated statistical theory for the variance of different things, which was to be used for estimating average staying time given entrances and exits (above). Some of the results are correct, some are not.
* `infer_time_distribution.py`: Tests various equations describing various aspects of the probability distributions of amount of time spent in the reservior, which can be used for inferring aspects of this probability distribution.
* `function_interpolator.py`: This file contains three classes each of which interpolates a
given 1D function as needed, in different ways.
Objects of all three classes can be called as if they were functions.
`DynamicBoundedInterpolator` is the fastest and most useful one.
  * `UnboundedInterpolator`: Slower for large `n` but simplest to use. It works by storing all the
  x-values that have been plugged into the function along with there corresponding y-values. When
  the y-value at a new x-value is requested, it checks to see if the closest two x-values on either
  side are close enough so that the requested value can be approximated with linear interpolation.
  If so, the interpolation is done. Otherwise, the function is evaluated and the new x and y pair is
  stored. Time complexity for calling the function is O(log(n)).
  * `BoundedInterpolator`: Evenly spaced x-values in a given interval and their corresponding y-values
  are stored. When the value at a given x-value in the interval is requested, the function value is
  approximated by linear interpolation. If function values at x below or above the interpolation
  range are requested, errors are thrown. Time complexity for calling the function is O(1).
  * `DynamicBoundedInterpolator`: Works the same as `BoundedInterpolator` except when a function
  value at x below or above the interpolation range is requested, the interpolation is expanded instead
of an error thrown. Time complexity for calling the function is O(1) (except when a value above the
  interpolation range is requested, in which case time complexity is linear in the difference between
  the requested x value and the interpolation range boundary).
