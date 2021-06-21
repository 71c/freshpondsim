from scipy import integrate
import numpy as np
import scipy.stats


class InOutTheory:
    def __init__(self, duration_dist, entrance_rate,
                 cumulative_entrance_rate=None):
        assert isinstance(duration_dist,
                          scipy.stats._distn_infrastructure.rv_frozen)
        self.T = duration_dist

        self.entrance_rate = np.vectorize(entrance_rate, otypes=[float])

        if cumulative_entrance_rate is None:
            def integral_func(t):
                if t < 0:
                    return 0
                y, abserr = integrate.quad(entrance_rate, 0, t)
                return y
            self.cumulative_entrance_rate = np.vectorize(integral_func, otypes=[float])
            self._closedform_cumulative_entrance_rate = False
        else:
            self.cumulative_entrance_rate = np.vectorize(cumulative_entrance_rate, otypes=[float])
            self._closedform_cumulative_entrance_rate = True

        self.lamda = self.entrance_rate
        self.Lamda = self.cumulative_entrance_rate

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
            lambda u: self.T.sf(u) * self.entrance_rate(t - u), 0, t)
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
            lambda u: self.T.pdf(u) * self.entrance_rate(t - u), 0, t)
        return y

    def cumulative_exit_rate(self, t):
        return self.cumulative_entrance_rate(t) - self.expected_n_people(t)

    def n_people_rv(self, t):
        return scipy.stats.poisson(self.expected_n_people(t))

    def duration_so_far_density(self, t):
        n_t = self.expected_n_people(t)
        return np.vectorize(lambda x: 0 if x < 0 else
            self.entrance_rate(t - x) * self.T.sf(x) / n_t)

    def expected_duration_so_far(self, t):
        assert t >= 0
        if t == 0:
            return 0.0
        n_t = self.expected_n_people(t)
        y, abserr = integrate.quad(
            lambda x: x * self.entrance_rate(t - x) * self.T.sf(x), 0, t)
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
            return self.T.pdf
        n_t = self.expected_n_people(t)
        def pdf(x):
            return self.T.pdf(x) * self.expected_entrances_in_interval(t - x, t) / n_t
        return pdf

    def expected_duration_observed(self, t):
        assert t >= 0
        if t == 0:
            return self._mean_duration
        pdf = self.duration_observed_density(t)
        y, abserr = integrate.quad(lambda x: x * pdf(x), 0, np.inf)
        return y

    def duration_so_far_steady_state_density(self, x):
        return self.T.sf(x) / self._mean_duration

    def duration_observed_steady_state_density(self, x):
        return x * self.T.pdf(x) / self._mean_duration

    def duration_observed_steady_state_mean(self):
        return self._mean_duration_observed

    def duration_so_far_steady_state_mean(self):
        return self._time_constant

    def change_in_n_people_rv(self, t1, t2):
        dt = t2 - t1
        integrand = lambda u: self.T.sf(u) * self.entrance_rate(t2 - u)
        nlambda, abserr = integrate.quad(integrand, 0, dt)
        tmp, abserr = integrate.quad(integrand, dt, t2)
        ne = self.expected_n_people(t1) - tmp
        return scipy.stats.skellam(nlambda, ne)


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

    iot = InOutTheory(duration_dist, entrance_rate, entrance_rate_integral)

    tvals = np.linspace(0, 60*24 * 2, num=100)

    tvals_hours = tvals / 60

    pr = cProfile.Profile()
    pr.enable()

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

    pr.disable()
    pr.print_stats(sort='cumulative')

    plt.show()
