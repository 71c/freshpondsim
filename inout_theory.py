from scipy import integrate
import numpy as np
import scipy.stats
from function_interpolator import DynamicBoundedInterpolator
from freshpondsim import random_times
import bisect


class InOutTheory:
    def __init__(self, duration_dist, entrance_rate,
                 cumulative_entrance_rate=None,
                 cumulative_entrance_rate_inverse=None,
                 interpolate_entrance_rate=False,
                 interpolate_cumulative_entrance_rate=False,
                 interpolate_duration_pdf=False,
                 interpolate_duration_sf=False,
                 time_res=None, duration_res=None):
        '''LOL don't pay attention to all the interpolate stuff
        since it doesn't even work anyway if you want to do the numerical
        integration because scipy.integrate.quad gets angry and there should be
        a way to make it work but I'm lazy so it doesn't work yet'''

        assert isinstance(duration_dist,
                          scipy.stats._distn_infrastructure.rv_frozen)

        if interpolate_entrance_rate or interpolate_cumulative_entrance_rate:
            if time_res is None:
                raise ValueError("Specify time_res for interpolation of entrance rate or cumulative entrance rate")
        
        if interpolate_duration_pdf or interpolate_duration_sf:
            if duration_res is None:
                raise ValueError("Specify duration_res for interpolation of duration pdf or sf")

        self.T = duration_dist

        self.cumulative_entrance_rate_inverse = cumulative_entrance_rate_inverse

        if interpolate_entrance_rate:
            self.entrance_rate = DynamicBoundedInterpolator(entrance_rate,
                        x1=0, x2=0, resolution=time_res, x_min=0, x_max=None)
        else:
            self.entrance_rate = np.vectorize(entrance_rate, otypes=[float])

        if cumulative_entrance_rate is None:
            def integral_func(t):
                if t < 0:
                    return 0
                y, abserr = integrate.quad(entrance_rate, 0, t)
                return y
            if interpolate_cumulative_entrance_rate:
                self.cumulative_entrance_rate = DynamicBoundedInterpolator(integral_func, x1=0, x2=0, resolution=time_res, x_min=0, x_max=None)
            else:
                self.cumulative_entrance_rate = np.vectorize(integral_func, otypes=[float])
            self._closedform_cumulative_entrance_rate = False
        else:
            if interpolate_cumulative_entrance_rate:
                self.cumulative_entrance_rate = DynamicBoundedInterpolator(cumulative_entrance_rate, x1=0, x2=0, resolution=time_res, x_min=0, x_max=None)
            else:
                self.cumulative_entrance_rate = np.vectorize(cumulative_entrance_rate, otypes=[float])
            self._closedform_cumulative_entrance_rate = True

        self.lamda = self.entrance_rate
        self.Lamda = self.cumulative_entrance_rate

        if interpolate_duration_pdf:
            self._pdf = DynamicBoundedInterpolator(self.T.pdf, x1=0, x2=0, resolution=duration_res, x_min=0, x_max=None)
        else:
            self._pdf = self.T.pdf
        
        if interpolate_duration_sf:
            self._sf = DynamicBoundedInterpolator(self.T.sf, x1=0, x2=0, resolution=duration_res, x_min=0, x_max=None)
        else:
            self._sf = self.T.sf

        self._mean_duration = self.T.mean()
        self._mean_square_duration = self.T.var() + self._mean_duration**2

        self._mean_duration_observed = self._mean_square_duration / self._mean_duration
        self._time_constant = self._mean_duration_observed / 2

        # vectorize these functions so we can apply them to numpy arrays
        # The only reason I do this is because it makes code elegant and simple.
        self.expected_n_people = np.vectorize(self.expected_n_people, otypes=[float])
        self.exit_rate = np.vectorize(self.exit_rate, otypes=[float])
        self.expected_duration_so_far = np.vectorize(self.expected_duration_so_far, otypes=[float])
        self.expected_duration_observed = np.vectorize(self.expected_duration_observed, otypes=[float])
        self.approx_expected_n_people_2 = np.vectorize(self.approx_expected_n_people_2, otypes=[float])
        self.approx_expected_n_people_3 = np.vectorize(self.approx_expected_n_people_3, otypes=[float])

    def expected_n_people(self, t):
        if t < 0:
            return 0.0
        y, abserr = integrate.quad(
            lambda u: self._sf(u) * self.entrance_rate(t - u), 0, t)
        return y

    def approx_expected_n_people_1(self, t):
        return self._mean_duration * self.entrance_rate(t - self._time_constant)

    def approx_expected_n_people_2(self, t):
        return self.expected_entrances_in_interval(t - self._mean_duration, t)

    def approx_expected_n_people_3(self, t):
        k1 = self._time_constant - self._mean_duration/2
        k2 = k1 + self._mean_duration
        return self.expected_entrances_in_interval(t - k2, t - k1)

    def exit_rate(self, t):
        if t < 0:
            return 0.0
        y, abserr = integrate.quad(
            lambda u: self._pdf(u) * self.entrance_rate(t - u), 0, t)
        return y

    def cumulative_exit_rate(self, t):
        return self.cumulative_entrance_rate(t) - self.expected_n_people(t)

    def n_people_rv(self, t):
        return scipy.stats.poisson(self.expected_n_people(t))

    def duration_so_far_density(self, t):
        assert t >= 0
        n_t = self.expected_n_people(t)
        return np.vectorize(lambda x: 0 if x < 0 else
            self.entrance_rate(t - x) * self._sf(x) / n_t)
    
    def duration_after_density(self, t):
        assert t >= 0
        n_t = self.expected_n_people(t)
        def pdf(x):
            y, abserr = integrate.quad(
                lambda u: self.entrance_rate(t - u) * self._pdf(x + u), 0, t)
            return y / n_t
        return pdf

    def expected_duration_so_far(self, t):
        assert t >= 0
        if t == 0:
            return 0.0
        n_t = self.expected_n_people(t)
        y, abserr = integrate.quad(
            lambda x: x * self.entrance_rate(t - x) * self._sf(x), 0, t)
        return y / n_t

    def expected_entrances_in_interval(self, t1, t2):
        if self._closedform_cumulative_entrance_rate:
            return self.Lamda(t2) - self.Lamda(t1)
        if t1 <= 0:
            return self.Lamda(t2)
        y, abserr = integrate.quad(self.entrance_rate, t1, t2)
        return y

    def duration_observed_density(self, t):
        assert t >= 0
        if t == 0:
            return self._pdf
        n_t = self.expected_n_people(t)
        def pdf(x):
            return self._pdf(x) * self.expected_entrances_in_interval(t - x, t) / n_t
        return pdf

    def expected_duration_observed(self, t):
        assert t >= 0
        if t == 0:
            return self._mean_duration
        pdf = self.duration_observed_density(t)
        y, abserr = integrate.quad(lambda x: x * pdf(x), 0, np.inf)
        return y

    def duration_so_far_steady_state_density(self, x):
        return self._sf(x) / self._mean_duration

    def duration_observed_steady_state_density(self, x):
        return x * self._pdf(x) / self._mean_duration

    def duration_observed_steady_state_mean(self):
        return self._mean_duration_observed

    def duration_so_far_steady_state_mean(self):
        return self._time_constant

    def change_in_n_people_rv(self, t1, t2):
        dt = t2 - t1
        integrand = lambda u: self._sf(u) * self.entrance_rate(t2 - u)
        nlambda, abserr = integrate.quad(integrand, 0, dt)
        tmp, abserr = integrate.quad(integrand, dt, t2)
        ne = self.expected_n_people(t1) - tmp
        return scipy.stats.skellam(nlambda, ne)

    def cumulative_n_people(self, t):
        if self._closedform_cumulative_entrance_rate:
            y, abserr = integrate.quad(lambda u: self._sf(u) * self.Lamda(t - u), 0, t)
            return y
        return self.n_integral(0, t)

    def n_integral(self, t1, t2):
        if self._closedform_cumulative_entrance_rate:
            return self.cumulative_n_people(t2) - self.cumulative_n_people(t1)
        y, abserr = integrate.dblquad(lambda tau, u: self._sf(tau) * self.lamda(u - tau), t1, t2, lambda u: 0, lambda u: u)
        return y
    
    def random_entrance_times(self, end_time):
        cum_entrance_rate = self.cumulative_entrance_rate if self._closedform_cumulative_entrance_rate else None
        entrance_times = np.array(list(
            random_times(0, end_time, self.entrance_rate, cum_entrance_rate,
                         self.cumulative_entrance_rate_inverse)
        ))
        return entrance_times

    def sample_realization(self, end_time):
        entrance_times = self.random_entrance_times(end_time)
        exit_times = entrance_times + self.T.rvs(entrance_times.shape)
        return entrance_times, exit_times


class FastRepeatedCalls:
    def __init__(self, func, bufsize):
        assert bufsize >= 1
        self._func = func
        self._bufsize = bufsize
        self._generate()
    
    def _generate(self):
        self._buffer = self._func(self._bufsize)
        self._index = 0

    def getval(self):
        if self._index == self._bufsize:
            self._generate()
        ret = self._buffer[self._index]
        self._index += 1
        return ret
    
    def getvals(self, n):
        assert n >= 1
        if n == 1:
            return self.getval()
        
        if self._index != self._bufsize:
            n_available = self._bufsize - self._index
            if n <= n_available:
                ret = self._buffer[self._index:self._index + n]
                self._index += n
                return ret
            else:
                first_vals = self._buffer[self._index:self._index + n_available]
                self._index += n_available # i.e., self._index = self._bufsize
                second_vals = self._func(n - n_available)
                return np.concatenate((first_vals, second_vals))
        else:
            return self._func(n)


class InOutSimulation:
    def __init__(self, inout_theory, end_time):
        assert isinstance(inout_theory, InOutTheory)
        self.inout_theory = inout_theory
        self.end_time = end_time
        self.entrance_times = None
        self.exit_times = None
        self.durations = None
        self._is_empty = True

    def refresh_entrance_times(self):
        assert not self._is_empty
        self.entrance_times = self.inout_theory.random_entrance_times(self.end_time)
        self.exit_times = self.entrance_times + self.durations

    def refresh_durations(self):
        assert not self._is_empty
        self.durations = self.inout_theory.T.rvs(self.entrance_times.shape)
        self.exit_times = self.entrance_times + self.durations
    
    def refresh_durations_conditional_on_inclusion(self, t0, inclusions=None):
        if self._is_empty:
            self.refresh()
            return

        if inclusions is None:
            inclusions = self.get_people_inclusions(t0)

        entrance_times = self.entrance_times

        # for all i < gt_t0_index, entrance_times[i] <= t0, and
        # for all i >= gt_t0_index, entrance_times[i] > t0.
        gt_t0_index = bisect.bisect(entrance_times, t0)
        assert np.all(entrance_times[:gt_t0_index] <= t0)
        assert np.all(entrance_times[gt_t0_index:] > t0)

        assert np.all(~inclusions[gt_t0_index:])

        inclusions_before = inclusions[:gt_t0_index]

        # T = self.inout_theory.T
        # T_fast = FastRepeatedCalls(T.rvs, gt_t0_index)
        # for i in range(gt_t0_index):
        #     T_threshold = t0 - entrance_times[i]
        #     T_i = T_fast.getval()
        #     inclusion = inclusions[i]
        #     while not ((T_i > T_threshold) == inclusion):
        #         T_i = T_fast.getval()
        #     self.durations[i] = T_i

        T = self.inout_theory.T
        T_fast = FastRepeatedCalls(T.rvs, gt_t0_index)
        self.durations[:gt_t0_index] = T.rvs(len(self.durations[:gt_t0_index]))
        T_thresholds = t0 - entrance_times[:gt_t0_index]
        curr_inclusions = self.durations[:gt_t0_index] > T_thresholds
        not_right = curr_inclusions != inclusions_before
        n_left = len(self.durations[:gt_t0_index][not_right])
        while n_left != 0:
            self.durations[:gt_t0_index][not_right] = T_fast.getvals(n_left)
            curr_inclusions = self.durations[:gt_t0_index] > T_thresholds
            not_right = curr_inclusions != inclusions_before
            n_left = len(self.durations[:gt_t0_index][not_right])

        self.durations[gt_t0_index:] = T.rvs(len(self.durations[gt_t0_index:]))

        self.exit_times = self.entrance_times + self.durations

        assert np.all(self.get_people_inclusions(t0) == inclusions)

    def refresh(self):
        self.entrance_times = self.inout_theory.random_entrance_times(self.end_time)
        self.durations = self.inout_theory.T.rvs(self.entrance_times.shape)
        self.exit_times = self.entrance_times + self.durations
        self._is_empty = False

    def get_people_inclusions(self, t):
        return (self.entrance_times <= t) & (t < self.exit_times)
    
    def n_people(self, t):
        return sum(self.get_people_inclusions(t))
    
    def get_entrance_times(self, t):
        return self.entrance_times[self.get_people_inclusions(t)]
    
    def get_exit_times(self, t):
        return self.exit_times[self.get_people_inclusions(t)]
    
    def get_durations(self, t):
        return self.durations[self.get_people_inclusions(t)]
    
    def get_residual_times_after(self, t):
        return self.get_exit_times(t) - t
    
    def get_residual_times_before(self, t):
        return t - self.get_entrance_times(t)
    
    def total_residual_time_after(self, t):
        return np.sum(self.get_residual_times_after(t))

    def total_residual_time_before(self, t):
        return np.sum(self.get_residual_times_before(t))

    def get_entrance_times_in_interval(self, t1, t2):
        return self.entrance_times[(t1 < self.entrance_times) & (self.entrance_times <= t2)]

    def get_exit_times_in_interval(self, t1, t2):
        return self.exit_times[(t1 < self.exit_times) & (self.exit_times <= t2)]

    def get_n_integral(self, t1, t2):
        entrances = self.get_entrance_times_in_interval(t1, t2)
        exits = self.get_exit_times_in_interval(t1, t2)
        return self.n_people(t1) * (t2 - t1) + np.sum(t2 - entrances) - np.sum(t2 - exits)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cProfile
    from tictoc import *
    
    entrance_rate_constant = 2

    ### Sinusoidal Entry Rate
    # a = 0.95 * entrance_rate_constant
    # period = 60*24
    # freq = 1/period
    # omega = 2*np.pi * freq
    # def entrance_rate(q):
    #     if q < 0:
    #         return 0.0
    #     return entrance_rate_constant + a * np.cos(omega * q)
    # def entrance_rate_integral(q):
    #     if q < 0:
    #         return 0.0
    #     return entrance_rate_constant * q + a / omega * np.sin(omega * q)

    # ### Constant Entry Rate
    # def entrance_rate(t):
    #     if t < 0:
    #         return 0
    #     return entrance_rate_constant
    # def entrance_rate_integral(t):
    #     if t < 0:
    #         return 0
    #     return entrance_rate_constant * t

    ### Linearly Increasing/Decreasing Entry Rate
    # lambda_increase_rate = 0.02
    # def entrance_rate(t):
    #     if t < 0:
    #         return 0
    #     return entrance_rate_constant + lambda_increase_rate * t
    # def entrance_rate_integral(t):
    #     if t < 0:
    #         return 0
    #     return entrance_rate_constant * t + 0.5 * lambda_increase_rate * t**2

    from simulation_defaults import _get_default_day_rate_func
    entrance_rate = _get_default_day_rate_func()
    entrance_rate_integral = None

    scale = 42.59286560661815 # scale parameter of Weibull distribution
    k = 1.5513080437971483 # shape parameter of weibull distribution
    duration_dist = scipy.stats.weibull_min(k, scale=scale)

    iot = InOutTheory(duration_dist, entrance_rate, entrance_rate_integral,
                        interpolate_duration_pdf=False, interpolate_duration_sf=False,
                        interpolate_entrance_rate=False, interpolate_cumulative_entrance_rate=False)

    tvals = np.linspace(0, 60*24 * 2, num=400)
    # tvals = np.linspace(0, 500, num=40)

    tvals_hours = tvals / 60

    # pr = cProfile.Profile()
    # pr.enable()
    # tic()

    plt.plot(tvals_hours, iot.expected_n_people(tvals), label='$n(t)$ expected num people')
    # approximations to n
    # plt.plot(tvals_hours, iot.approx_expected_n_people_1(tvals), label='$n_{approx1}(t)$')
    # plt.plot(tvals_hours, iot.approx_expected_n_people_2(tvals), label='$n_{approx2}(t)$')
    # plt.plot(tvals_hours, iot.approx_expected_n_people_3(tvals), label='$n_{approx3}(t)$')
    plt.xlabel('time (hours)')
    plt.ylabel('people')
    plt.legend()

    plt.figure()
    plt.plot(tvals_hours, iot.entrance_rate(tvals), label='$\\lambda(t)$ entrance rate')
    plt.plot(tvals_hours, iot.exit_rate(tvals), label='$e(t)$ exit rate')
    plt.xlabel('time (hours)')
    plt.ylabel('rate (people / minute)')
    plt.legend()

    plt.figure()
    plt.plot(tvals_hours, iot.expected_duration_so_far(tvals), label='expected duration so far')
    o = iot.expected_duration_observed(tvals)
    plt.plot(tvals_hours, o, label='expected duration observed')
    # plt.plot(tvals, o / iot._mean_duration, label='expected duration observed / mean duration')
    plt.plot(tvals_hours, iot.duration_so_far_steady_state_mean()*np.ones(tvals.shape), label='expected duration so far steady state')
    plt.plot(tvals_hours, iot.duration_observed_steady_state_mean()*np.ones(tvals.shape), label='expected duration observed steady state')
    plt.plot(tvals_hours, iot._mean_duration*np.ones(tvals.shape), label='mean duration')
    plt.xlabel('time (hours)')
    plt.ylabel('duration (minutes)')
    plt.legend()

    # tocl()
    # pr.disable()
    # pr.print_stats(sort='tottime')

    plt.show()
