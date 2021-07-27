import numpy as np
from numpy.random import default_rng
from poibin import PoiBin
from tqdm import tqdm


rng = default_rng()


def random_bernoulli_vectors(probs, n=1):
    k = len(probs)
    probs = np.array(probs)

    if n == 1:
        return (rng.random(k) <= probs).astype(int)
    return (rng.random((n, k)) <= probs[None, :]).astype(int)


def random_bernoulli_vectors_conditional_on_sum(probs, s, n=1, do_tqdm=False):
    lists = []
    total = 0
    if do_tqdm:
        pbar = tqdm(total=n)
    while total < n:
        rvs = random_bernoulli_vectors(probs, n=2*n) # can change number generated
        conditional_rvs = rvs[np.sum(rvs, axis=1) == s]
        total += conditional_rvs.shape[0]
        if do_tqdm:
            pbar.update(conditional_rvs.shape[0])
        lists.append(conditional_rvs)
    if do_tqdm:
        pbar.close()
    ret = np.vstack(lists)[:n]
    return ret[0] if n == 1 else ret


def get_poibin_pmf_RF1(p):
    n = len(p)
    A = np.zeros(n + 1, dtype=np.longdouble)
    A[0] = 1.0
    for j in range(0, n):
        for i in range(j+1, -1, -1):
            a_im1 = 0.0 if i == 0 else A[i-1]
            A[i] += (a_im1 - A[i]) * p[j]
    return A


def get_poibin_pmf(p, use_fft=False):
    if use_fft:
        return PoiBin(p).pmf_list
    return get_poibin_pmf_RF1(p)


def get_bernoulli_probs_conditional_on_sum(probs, use_fft=True):
    '''If X_1,...,X_N are Bernoulli random variables,
    let S = X_1 + ... + X_N be their sum. Suppose you know their sum equals s.
    This function calculates P(X_i = 1 | S = s) for all values i, and for all s.
    This is computed using Bayes' theorem.
    
    Arguments:
        probs: the probabilities of success
        use_fft: whether to use FFT method (otherwise, RF1 method is used).
                FFT is faster but RF1 is more accurate.
                For large N, FFT tends to be accurate in a certain range of s,
                but it is extremely unlikely for s to not be in this range
                anyway. The RF1 method tends to be correct for a wider range of
                s, but it, to, can give out as well.
    Returns:
        A matrix X with the probabilities explained above.
        Each row corresponds to a different s, and s = 1...N.
        Each column corresponds to i. i = 1...N.
        We have that (with i starting from 1):
                P(X_i=1 | S=s) == X[s-1, i-1].
    '''
    n_rvs = len(probs)
    probs = np.array(probs)
    assert all((0 < probs) & (probs <= 1))

    # P(S = s)
    pmf = get_poibin_pmf(probs, use_fft=use_fft)

    # For all i, this array will contain P(S=s | X_i = 1)
    # Each row corresponds to s. s = 1...N (s cannot be 0; we condition X_i=1)
    # Each column corresponds to i. i = 1...N
    probs_of_sum_given_Xi = np.empty((n_rvs, n_rvs))
    for i in range(n_rvs):
        new_probs = np.concatenate((probs[:i], probs[i+1:]))
        probs_of_sum_given_Xi[:, i] = get_poibin_pmf(new_probs, use_fft=use_fft)

    return probs_of_sum_given_Xi * probs / pmf[1:, None]


if __name__ == "__main__":
    from tictoc import *

    # tic()
    # rvs = random_bernoulli_vectors([0.3, 0.9, 0.4, 0.5, 0.04], 100_000)
    # toc()

    # tic()
    # rvs = random_bernoulli_vectors_conditional_on_sum([0.3, 0.7, 0.4, 0.5, 0.04, 0.9], s=5, n=100_000)
    # toc()
    # print(rvs)
    # print(rvs.mean(axis=0))

    # print(rvs)

    # print(rvs.shape[0])

    p = [0.3, 0.7, 0.4, 0.5, 0.04, 0.9]
    s = 5

    for _ in range(20):
        print(p)
        X = get_bernoulli_probs_conditional_on_sum(p)
        p = X[s-1]
