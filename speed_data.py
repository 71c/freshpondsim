import matplotlib.pyplot as plt
import numpy as np

def group_by(items, group_function=lambda x: x):
    """
    Groups items by a grouping function
    parameters:
        items: iterable containing things
        group_function: function that gives the same result when called on two
            items in the same group
    returns: a dict where the keys are results of the function and values are
        lists of items that when passed to group_function give that key
    """
    group_dict = {}
    for item in items:
        value = group_function(item)
        if value in group_dict:
            group_dict[value].append(item)
        else:
            group_dict[value] = [item]
    return group_dict


times_1 = [(5.99, 'bike'), (25.03, 'walk'), (17.65, 'walk'), (9.71, 'run'),
           (21.73, 'walk'), (4.48, 'bike'), (13.12, 'run'), (12.43, 'walk'),
           (19.12, 'walk'), (18.32, 'walk'), (26.70, 'walk'), (8.37, 'run')]
distance_1 = 81 + 5 / 12

times_2 = [(16.46, 'walk'), (10.46, 'run'), (10.33, 'run'), (9.88, 'bike'),
           (21.42, 'walk'), (18.0, 'walk'), (13.86, 'run'), (7.69, 'run'),
           (7.30, 'run'), (9.16, 'run'), (7.84, 'run'), (21.11, 'walk'),
           (8.35, 'run'), (8.68, 'run'), (18.55, 'walk'), (7.67, 'run'),
           (22.38, 'walk'), (27.08, 'walk'), (13.60, 'walk'), (8.69, 'run'),
           (18.98, 'walk'), (10.21, 'run')]
distance_2 = 80 + 10 / 12

# minutes per mile
paces_1 = [(88 * dt / distance_1, name) for dt, name in times_1]
paces_2 = [(88 * dt / distance_2, name) for dt, name in times_2]

# meters per second
# paces_1 = [(0.3048 * distance_1 / dt, name) for dt, name in times_1]
# paces_2 = [(0.3048 * distance_2 / dt, name) for dt, name in times_2]


paces = paces_1 + paces_2
paces_dict = {
    k: [x[0] for x in v]
    for k, v in group_by(paces, lambda x: x[1]).items()
}

# plt.hist([x[0] for x in paces], bins='auto')
# plt.show()

# minutes per mile == 26.8224 / (meters per second)

for name, values in paces_dict.items():

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    mean_standard_error = std / np.sqrt(n)
    print(f"{name}: n: {n}, mean: {mean:.4g} ± {mean_standard_error:.4g}, std: {std:.4g}")

    plt.hist(values, bins='auto', density=True, histtype='step', label=name)
plt.legend()

plt.show()
