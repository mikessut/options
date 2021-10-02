from options.garam import *
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


def plot_vs_std_norm(y, bins=50):
    hist = np.histogram(y, bins=bins)
    delta = np.diff(hist[1])[0]
    centers = (hist[1][:-1] + hist[1][1:]) / 2
    x = np.linspace(-3, 3, 100)
    plt.semilogy(x, norm.pdf(x))
    plt.semilogy(centers, hist[0] / len(y) / delta, 'o')


def test_normalization():
    df = pd.read_pickle('eth_historical.pd')
    df['lr'] = np.log(df.Close / df.Close.shift())
    df['r2'] = df.lr**2
    df = df.dropna()

    plt.figure()
    idx = np.abs(df.r2) != 0 # > .00001
    x = np.log(df[idx].r2)
    x = (x-x.mean())/x.std()
    plot_vs_std_norm(x)

    X = fit_hammer_params(df.r2)
    print(X)
    mean_offset, p1, p2, p3 = X.x

    idx = np.abs(df.r2) > .0000001
    x = np.log(df.r2[idx]) + mean_offset
    y = normalize_hammer(x, p1, p2, p3)

    df.loc[idx, 'x'] = x
    df.loc[idx, 'y'] = y

    plt.figure()
    plot_vs_std_norm(y / y.std())
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

    print(acf(Y, nlags=8, fft=True))


def test_sim_xcorr():

    A, B, L = simulate_xcorr([1, .5], [1, -.5], 2000)
    print(L)

    plt.xcorr(A, B)
    return A, B, L
