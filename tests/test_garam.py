from options.garam import *
from options import garam
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, ccf
from scipy.stats import pearsonr
import pytest


def plot_vs_std_norm(y, bins=50):
    hist = np.histogram(y, bins=bins)
    delta = np.diff(hist[1])[0]
    centers = (hist[1][:-1] + hist[1][1:]) / 2
    x = np.linspace(-3, 3, 100)
    plt.semilogy(x, norm.pdf(x))
    plt.semilogy(centers, hist[0] / len(y) / delta, 'o')


def test_normalization():
    df = pd.read_pickle('tests/eth_historical.pd')
    df['lr'] = np.log(df.Close / df.Close.shift())
    df['r2'] = df.lr**2
    df = df.dropna()

    plt.figure()
    idx = np.abs(df.r2) != 0 # > .00001
    df.loc[np.abs(df.r2) == 0, 'r2'] = .000001
    x = np.log(df.r2)
    x = (x-x.mean())/x.std()
    plot_vs_std_norm(x)

    X = fit_hammer_params(df.r2)
    print(X)
    mean_offset, p1, p2, p3 = X.x

    x = np.log(df.r2) + mean_offset  # zero mean, std = 2.6
    y = normalize_hammer(x, p1, p2, p3)  # zero mean, std = 6.6

    df.loc[:, 'x'] = x
    df.loc[:, 'y'] = y

    plt.figure()
    plot_vs_std_norm(y / y.std())

    # Correlations
    plot_acf(y)
    plt.title('y - normed vol')

    plot_acf(df.lr)
    plt.title('z (aka lr)')

    plt.figure()
    plt.xcorr(df.lr, y)

    # similar xcorr plot
    plt.figure()
    lags = np.arange(-9, 10)
    plt.stem(lags, [np.corrcoef(df.lr, np.roll(y, n))[0, 1] for n in lags])
    # equivalently [pearsonr(df.lr, np.roll(y, n))[0] for n in lags]

    return df, X


def test_sim_auto_corr():
    df = pd.read_pickle('tests/eth_historical.pd')
    df['lr'] = np.log(df.Close / df.Close.shift())
    df['r2'] = df.lr**2
    idx = np.abs(df.r2) > .0000001
    lr2 = np.log(df[idx].r2)

    rhos = acf(lr2, nlags=8, fft=True)

    print(rhos)

    Y = simulate_auto_corr(rhos, 3000)
    print(Y.std())

    print(acf(Y, nlags=8, fft=True))


def test_xcorr_mat():
    rho_g = [1, .5, .2]
    rho_h = [1, .9, .8]
    rho_forward = [.5, .6, .1]
    rho_backward = [.5, .4, .3]
    cov_mat = garam._xcorr_cov_mat(rho_g, rho_h, rho_forward, rho_backward)
    print(cov_mat)
    
    np.testing.assert_array_equal(np.diag(cov_mat), 1)
    for rho, idx in zip([rho_g, rho_h],
                        [(slice(3), slice(3)),
                         (slice(3, 6), slice(3, 6))]):
        for n in range(1, 3):
            np.testing.assert_array_equal(np.diag(cov_mat[idx], n), rho[n])
            np.testing.assert_array_equal(np.diag(cov_mat[idx], -n), rho[n])
    # UR
    idx = (slice(3), slice(3, 6))
    np.testing.assert_array_equal(np.diag(cov_mat[idx]), rho_forward[0])
    for n in range(1, 3):
        np.testing.assert_array_equal(np.diag(cov_mat[idx], n), rho_forward[n])
        np.testing.assert_array_equal(np.diag(cov_mat[idx], -n), rho_backward[n])

    idx = (slice(3, 6), slice(3))
    np.testing.assert_array_equal(np.diag(cov_mat[idx]), rho_forward[0])
    for n in range(1, 3):
        np.testing.assert_array_equal(np.diag(cov_mat[idx], -n), rho_forward[n])
        np.testing.assert_array_equal(np.diag(cov_mat[idx], n), rho_backward[n])
    
    # Test Symmetric
    idx = np.triu_indices_from(cov_mat, 1)
    np.testing.assert_equal(cov_mat[idx],
                            cov_mat[idx[1], idx[0]])

def test_sim_xcorr():

    # df = pd.read_pickle('tests/eth_historical.pd')
    # df['lr'] = np.log(df.Close / df.Close.shift())
    # df['r2'] = df.lr**2
    # df = df.dropna()
    # 
    # plt.figure()
    # idx = np.abs(df.r2) != 0 # > .00001
    # x = np.log(df[idx].r2)
    # x = (x-x.mean())/x.std()
    # plt.xcorr(x, df[idx].lr)
    # 
    # return ccf(x, df[idx].lr)

    x = np.arange(1000)
    r = np.random.randn(len(x))
    v = -.5 * r + np.random.randn(len(x)) + np.roll(r, 1) * -.2 + np.roll(r, -1) * .2
    rho_backward = [np.corrcoef(r, np.roll(v, n))[0, 1] for n in range(2)]
    rho_forward = [np.corrcoef(r, np.roll(v, n))[0, 1] for n in range(0, -2, -1)]

    AB = simulate_xcorr([1, 0, 0], [1, 0, 0], rho_forward + [0,], rho_backward + [0,], len(x))
    generated_rho_backward = [np.corrcoef(AB[0, :], np.roll(AB[1, :], n))[0, 1] for n in range(2)]
    generated_rho_forward = [np.corrcoef(AB[0, :], np.roll(AB[1, :], n))[0, 1] for n in range(0, -2, -1)]
    np.testing.assert_allclose(rho_forward, generated_rho_forward, atol=.1)
    np.testing.assert_allclose(rho_backward, generated_rho_backward, atol=.1)
