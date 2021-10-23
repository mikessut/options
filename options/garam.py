"""
Inspired by this question:
https://quant.stackexchange.com/questions/21842/calibrating-stochastic-volatility-model-from-price-history-not-option-prices

Which leads to this paper on GARAM model:
http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1428555

More recent paper from Vivek Kapoor:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3761304
"""
from operator import le
import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.optimize import least_squares, root_scalar


def normalize_hammer(x, p1, p2, p3):
    """
    Used to make the log squared returns normal.
    """
    return x * (1 + p1 * (np.pi / 2 + np.arctan(p2 * (x + p3))))


def inv_hammer(y, p1, p2, p3):
    soln = root_scalar(lambda x: y - normalize_hammer(x, p1, p2, p3), x0=0, x1=1)
    if soln.converged:
        return soln.root


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


def _seed_predictors(g_seeds, h_seeds, L):
    """
    
    +-    -+   +-                        -+ +-    -+
    |  g   |   | l                        | |  u   |
    |   -1 |   |  00                      | |   -1 |
    |      |   |                          | |      |
    |  g   |   | l   l                    | |  u   |
    |   0  |   |  10  11                  | |   0  |
    |      |   |                          | |      |
    |  g   |   | l   l   l                | |  u   |
    |   1  |   |  20  21  22              | |   1  |
    |      | = |                          | |      |
    |  h   |   | l   l   l   l            | |  v   |
    |   -1 |   |  30  31  32  33          | |   -1 |
    |      |   |                          | |      |
    |  h   |   | l   l   l   l   l        | |  v   |
    |   0  |   |  40  41  42  43  44      | |   0  |
    |      |   |                          | |      |
    |  h   |   | l   l   l   l   l   l    | |  v   |
    |   1  |   |  50  51  52  53  54  55  | |   1  |
    +-    -+   +-                        -+ +-    -+

    Unknowns up1, u0, vp1, v0, g1, h1
    Random: u1, v1

    g   = u
     -1    -1

    l   *  g   + l   * u  = g
     10     -1    11    0    0

    u0 = (g0 - l10*gp1) / l11

    u1 = (g1 - l20*gp1 - l21*g0) / l22
    """
    lrho = L.shape[0] // 2
    Linv = np.linalg.inv(L)
    up = Linv[:lrho//2+1, :lrho//2+1] @ np.vstack(g_seeds)
    u = np.vstack([up, np.random.randn(lrho // 2)])
    g = L[:lrho, :lrho] @ u
    vp = Linv[lrho:lrho+lrho//2+1, :lrho+lrho//2+1] @ np.vstack([g, np.vstack(h_seeds)])    
    v = np.vstack([vp, np.random.randn(lrho // 2)])
    return u, v


def simulate_xcorr(rho_g, rho_h, rho_forward, rho_backward, num_samples,
                   g_seeds=None, h_seeds=None):
    """
    :param g_seeds: Realized data in chronological order
    :param h_seeds: Realized data in chronological order

    """
    assert len(rho_g) % 2 == 1, "Length of correlations must be odd"
    cov_mat = _xcorr_cov_mat(rho_g, rho_h, rho_forward, rho_backward)
    lrho = len(rho_g)
    
    L = np.linalg.cholesky(cov_mat)

    if g_seeds is not None:
        assert len(g_seeds) == len(rho_g)
        assert len(h_seeds) == len(rho_h)

        u, v


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


def sim_garam():
    """
    r = np.log(today / yesterday)
    x = np.log(r**2 / R**2) = np.log(r**2) + mean_offset
    y = x*(1+p1*(pi/2 + atan(p2*(x+p3))))

    z normal and correlated to y

    r = I * r
    I = sign(z)
    """
    mean_offset, p1, p2, p3 = (7.310869029875786,
                               1.4680161660760758,
                               0.15365361394057306,
                               -2.1170813844931735)
    y_std = 6.610029945063
    z_acorr = np.zeros((13, ))
    z_acorr[0] = 1
    y_acorr = [1., 0.15609107, 0.1217776 , 0.14139572, 0.11028372,
               0.10320274, 0.1252665 , 0.14323801, 0.07744252, 0.05458113,
               0.08284051, 0.07172931, 0.06604487]
    zy_forward = [0.058317939998810915,
                  0.042898529214825676,
                  0.006864844106183375,
                  0.007019942397415309,
                  0.0018073768467972565,
                  -0.0009463001559640069,
                  0.015057954783676279,
                  0.04183635088839025,
                  -0.015527960543866933,
                  -0.0003720180721489296,
                  -0.03151007791432902,
                  0.050994869800271264,
                  -0.011975136914581923]
    zy_backward = [0.058317939998810915,
                   -0.028555262080963626,
                   0.004473658237229506,
                   0.0020374398456827134,
                   0.0034062578251732104,
                   0.04092458378189857,
                   -0.006354430680190513,
                   -0.005450535011509913,
                   0.034374758133681536,
                   0.02857448662093088,
                   0.0010408983590319903,
                   -0.00996683831507115,
                   0.00666669951144906]
    
    AB = simulate_xcorr(z_acorr, y_acorr, zy_backward, zy_forward, 2000)
    z = AB[0, :]
    y = AB[1, :] * y_std
    x = np.vectorize(lambda y: inv_hammer(y, p1, p2, p3))(y)

    rsq = np.exp(x - mean_offset)
    print(f"rsq mean: {rsq.mean()}")
    r = np.sign(z) * np.sqrt(rsq)
    print(f"r mean: {r.mean()}")

    #f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(rsq)
    ax[1].plot(14 * np.exp(np.cumsum(r)))