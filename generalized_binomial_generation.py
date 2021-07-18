import numpy as np
from numpy.random import default_rng
from tictoc import *


rng = default_rng()


def random_bernoulli_vectors(probs, n=1):
    k = len(probs)
    probs = np.array(probs)

    if n == 1:
        return (rng.random(k) <= probs).astype(int)
    return (rng.random((n, k)) <= probs[None, :]).astype(int)


def random_bernoulli_vectors_conditional_on_sum(probs, s, n=1):
    lists = []
    total = 0
    while total < n:
        rvs = random_bernoulli_vectors(probs, n=2*n) # can change number generated
        conditional_rvs = rvs[np.sum(rvs, axis=1) == s]
        total += conditional_rvs.shape[0]
        lists.append(conditional_rvs)
    ret = np.vstack(lists)[:n]
    return ret[0] if n == 1 else ret


if __name__ == "__main__":
    # tic()
    # rvs = random_bernoulli_vectors([0.3, 0.9, 0.4, 0.5, 0.04], 100_000)
    # toc()

    tic()
    rvs = random_bernoulli_vectors_conditional_on_sum([0.3, 0.7, 0.4, 0.5, 0.04, 0.9], s=5, n=100_000)
    toc()
    print(rvs)
    print(rvs.mean(axis=0))

    # print(rvs)

    # print(rvs.shape[0])
