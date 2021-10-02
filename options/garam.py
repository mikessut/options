"""
Inspired by this question:
https://quant.stackexchange.com/questions/21842/calibrating-stochastic-volatility-model-from-price-history-not-option-prices

Which leads to this paper on GARAM model:
http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1428555
"""
from operator import le
import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.optimize import least_squares


def normalize_hammer(x, p1, p2, p3):
    """
    Used to make the log squared returns normal.
    """
    return x * (1 + p1 * (np.pi / 2 + np.arctan(p2 * (x + p3))))


def non_normal_errs(X, rsq):
    """
    Doesn't account for standard deviation being 1
    """
    mean_offset, p1, p2, p3 = X
    x = np.log(rsq) + mean_offset
    y = normalize_hammer(x, p1, p2, p3)
    # y = y / y.std()
    return [np.mean(y), kurtosis(y), skew(y), #moment(y/y.std(), 5)]
            ((y-y.mean())**5).mean() / y.std()**5]


def fit_hammer_params(rsq):
    idx = np.abs(rsq) > .000001
    idx = rsq != 0
    rsq = rsq[idx]
    X = least_squares(non_normal_errs,
                      [7.5, 0.5, .5, 2],
                      args=(rsq, ),
                      )
    return X


def simulate_auto_corr(rhos, num_samples):
    """
    I don't think this works in the general case -- i.e. can't create any
    autocorrelated data.
    """
    assert rhos[0] == 1.0
    cov_mat = np.eye(len(rhos))
    for n, p in enumerate(rhos[1:]):
        cov_mat += np.diag(np.ones((len(rhos)-n-1, )) * p, n+1)
        cov_mat += np.diag(np.ones((len(rhos)-n-1, )) * p, -n-1)
    
    L = np.linalg.cholesky(cov_mat)

    U = np.random.randn(num_samples)
    H = np.zeros((num_samples, ))

    for n in range(num_samples):
        if n < len(rhos):
            H[n] = L[n, :n+1].dot(U[:n+1])
        else:
            H[n] = L[-1, :].dot(U[n-(len(rhos)-1):n+1])
    return H


def simulate_xcorr(rho_forward, rho_backward, num_samples):
    """

    """
    assert rho_forward[0] == 1.0
    assert rho_backward[0] == 1.0
    assert len(rho_forward) == len(rho_backward)
    lrho = len(rho_forward)

    L = []
    for rhos in [rho_forward, rho_backward]:
        cov_mat = np.eye(len(rhos))
        for n, p in enumerate(rhos[1:]):
            cov_mat += np.diag(np.ones((len(rhos)-n-1, )) * p, n+1)
            cov_mat += np.diag(np.ones((len(rhos)-n-1, )) * p, -n-1)
        L.append(np.linalg.cholesky(cov_mat))

    U = np.random.randn(num_samples)
    #V = np.random.randn(num_samples)
    V = U
    A = np.zeros((num_samples, ))
    B = np.zeros((num_samples, ))

    for n in range(num_samples):
        if n < lrho:
            A[n] = L[0][n, :n+1].dot(U[:n+1])
            B[n] = L[1][-1, ::-1].dot(V[n:n+lrho])
        elif n > (num_samples - lrho):
            A[n] = L[0][-1, :].dot(U[n-(lrho-1):n+1])
            B[n] = L[1][num_samples-n-1, :num_samples-n].dot(V[n:])
        else:
            A[n] = L[0][-1, :].dot(U[n-(lrho-1):n+1])
            B[n] = L[1][-1, ::-1].dot(V[n:n+lrho])
    return A, B, L