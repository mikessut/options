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
    #print(L)
    U = np.zeros((len(rhos), num_samples))
    U[-1, :] = np.random.randn(num_samples)
    for n in range(1, len(rhos)):
        U[len(rhos) - n - 1, :] = np.roll(U[-1, :], n)

    #print(U)
    return L[-1, :].dot(U)
    # U = np.random.randn(num_samples)
    # H = np.zeros((num_samples, ))

    for n in range(num_samples):
        if n < len(rhos):
            H[n] = L[n, :n+1].dot(U[:n+1])
        else:
            H[n] = L[-1, :].dot(U[n-(len(rhos)-1):n+1])
    return H


def _xcorr_cov_mat(rho_g, rho_h, rho_forward, rho_backward):
    """
    Simulate both auto correlation and cross correlation.

    Example of the cross corr terms. (Autocorr terms not shown.)
        g      g     g
        -1     0     1

        h      h     h
        -1     0     1

    p        = [p     , p     , ...]  = [p  , p  , ...]
     forward     g0,h0   g0,h1            f0   f1

    p         = [p     , p      , ...] = [p  , p  , ...]
     backward    g0,h0   g0,h-1           b0   b1

    p   = p
    f0    b0


         g    g   g   h   h   h
          -1   0   1   -1  0   1
        ┌─────────────────────────
        │
    g   │             p   p    p
     -1 │              0   f1   f2
        │
    g   │             p   p    p
     0  │              b1  0    f1
        │
    g   │             p   p    p
     1  │              b2  b1   0
        │
    h   │ p   p   p
     -1 │  0   b1  b2
        │
    h   │ p   p   p
     0  │  f1  0   b1
        │  
    h   │ p   p   p
     1  │  f2  f1  0
    """
    assert rho_forward[0] == rho_backward[0]
    assert len(rho_forward) == len(rho_backward)
    assert len(rho_g) == len(rho_h)
    assert len(rho_g) == len(rho_forward)
    lrho = len(rho_forward)

    cov_mat = np.zeros((2*lrho, 2*lrho))

    # Autocorrelations
    for rhos, idx in zip([rho_g, rho_h], 
                         [(slice(0, lrho), slice(0, lrho)), 
                          (slice(lrho, 2*lrho), slice(lrho, 2*lrho))]):
        cov_mat[idx] = np.eye(lrho)
        for n, p in enumerate(rhos[1:]):
            cov_mat[idx] += np.diag(np.ones((len(rhos)-n-1, )) * p, n+1)
            cov_mat[idx] += np.diag(np.ones((len(rhos)-n-1, )) * p, -n-1)

    # Crosscorrelations
    idx_ur = (slice(0, lrho), slice(lrho, 2*lrho))
    idx_ll = (slice(lrho, 2*lrho), slice(0, lrho))

    cov_mat[idx_ur] += np.eye(lrho) * rho_forward[0]
    cov_mat[idx_ll] += np.eye(lrho) * rho_forward[0]
    for n in range(1, lrho):
        cov_mat[idx_ur] += np.diag(np.ones((lrho-n, )) * rho_forward[n], n)
        cov_mat[idx_ur] += np.diag(np.ones((lrho-n, )) * rho_backward[n], -n)
        cov_mat[idx_ll] += np.diag(np.ones((lrho-n, )) * rho_backward[n], n)
        cov_mat[idx_ll] += np.diag(np.ones((lrho-n, )) * rho_forward[n], -n)
    return cov_mat

def simulate_xcorr(rho_g, rho_h, rho_forward, rho_backward, num_samples):
    """

    """
    assert len(rho_g) % 2 == 1, "Length of correlations must be odd"
    cov_mat = _xcorr_cov_mat(rho_g, rho_h, rho_forward, rho_backward)
    lrho = len(rho_g)
    
    L = np.linalg.cholesky(cov_mat)

    U = np.random.randn(num_samples)
    V = np.random.randn(num_samples)

    GH = np.zeros((2 * lrho, num_samples))

    # Does lrho need to be odd? Not sure...
    GH[lrho // 2, :] = U
    GH[lrho + lrho // 2, :] = V

    for n in range(1, lrho//2+1):
        GH[lrho // 2 - n, :] = np.roll(GH[lrho // 2, :], n)
        GH[lrho // 2 + n, :] = np.roll(GH[lrho // 2, :], -n)
        GH[lrho + lrho // 2 - n, :] = np.roll(GH[lrho + lrho // 2, :], n)
        GH[lrho + lrho // 2 + n, :] = np.roll(GH[lrho + lrho // 2, :], -n)

    return L[[lrho // 2, lrho + lrho // 2], :].dot(GH)
