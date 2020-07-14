# freshpondsim
This is a simulation program that models people going around a reservoir,
or any long circular path, throughout time.

## Files
The main files are:
* `freshpondsim.py`: This is the file that contains the code for the simulation program.
It contains two classes: `FreshPondPedestrian` and `FreshPondSim`.
  * A `FreshPondPedestrian` object represents a person going around the reservoir.
  * A `FreshPondSim` object represents a simulation of people going around the reservoir.
    It cointains a list  of `FreshPondPedestrian`s stored in it called `pedestirans`.
    The number of people at time `t` is accessed by `self.n_people(t)`
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
