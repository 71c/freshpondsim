import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import expit, gamma


def parameterized_logistic_function(x0, h, a=0, b=1):
    """Returns a logistic function with a given starting value, ending value,
    middle, and width.
    x0: middle
    h: width
    a: start value
    b: end value"""
    return lambda x: a + (b - a) * expit((x - x0) / h)


def integrate(func, a, b):
    y, abserr = scipy.integrate.quad(func, a, b)
    return y


def get_conditional_pdf(stat_dist_pdf, stat_to_prob_func):
    const = integrate(
        lambda x: stat_dist_pdf(x) * stat_to_prob_func(x), -np.inf, np.inf)
    return lambda x: stat_dist_pdf(x) * stat_to_prob_func(x) / const


stat_dist = scipy.stats.norm(loc=142, scale=80)

# mean_stat = 142.0 # expected value of time spent
# k = 1.4 # shape parameter of weibull distribution
# scale = mean_stat / gamma(1 + 1/k) # scale parameter of Weibull distribution
# stat_dist = scipy.stats.weibull_min(k, scale=scale)


# stat_to_prob_function = parameterized_logistic_function(x0=100, h=17, a=0, b=2)
stat_to_prob_function = lambda x: (np.sin(x/20) + 1) / 2 * 10


n_samples = 200000
stats = stat_dist.rvs(n_samples)
probs = stat_to_prob_function(stats) * 0.02
uniform_samples = np.random.random(n_samples)
conditional_stats = stats[uniform_samples < probs]

conditional_pdf = get_conditional_pdf(stat_dist.pdf, stat_to_prob_function)

plt.hist(conditional_stats, bins='auto', density=True, label='conditional stats histogram')
x = np.linspace(min(conditional_stats), max(conditional_stats), 300)
plt.plot(x, conditional_pdf(x), label='conditional stats pdf')
plt.legend()
plt.show()


print(len(conditional_stats) / n_samples)
print(stat_dist.expect(stat_to_prob_function))
# print(np.mean(stats))
# print(np.mean(conditional_stats))



# print(stat_dist_samples)
# print(stat_to_prob_function(stat_dist_samples))
# print()
