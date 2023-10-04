from matplotlib.widgets import MultiCursor
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from options import egarch
from options import PutOption, CallOption
import pytest


# @pytest.mark.skip("plots")
def test_egarch():
    df = pd.read_csv(pathlib.Path(__file__).parent / "SPY_20171104-20221104.csv")
    df.index = pd.DatetimeIndex(df.Date) 
    lr = np.log(df.Close / df.Close.shift())[1:]

    params = egarch.EGARCHMonteCarlo.fit(lr)

    print(params)

    f, ax = plt.subplots(3, 1, sharex=True)
    
    ax[0].plot(df.Close)
    ax[1].plot(lr)
    ax[2].plot(pd.Series(np.sqrt(egarch.EGARCHMonteCarlo.calc_garch(lr, **params) * 252), lr.index), label='egarch')
    # multi = MultiCursor(f.canvas, ax, horizOn=True)

    # plt.show()
    plt.savefig("egarch_fit.png")


# @pytest.mark.skip("plots")
def test_egarch_iv():
    """
    Shows skew (higher IV for strikes to the downside)
    """
    df = pd.read_csv(pathlib.Path(__file__).parent / "SPY_20171104-20221104.csv")
    df.index = pd.DatetimeIndex(df.Date) 
    lr = np.log(df.Close / df.Close.shift())[1:]

    params = egarch.EGARCHMonteCarlo.fit(lr)
    print(f"Fit to historical data: {params}")

    p = 102.1
    dte = 50
    K = [90, 95, 100, 105, 110]
    steps_per_year = 252
    r = .015

    np.random.seed(1)
    g = egarch.EGARCHMonteCarlo(p, 0 * -0.02, K, dte, .2**2 / 252, **params, r=r, num_sims=200000, days_in_year=steps_per_year)
    g.run()

    ivs = np.zeros((len(K), 2))

    for n, k in enumerate(K):
        put = PutOption(k, dte / steps_per_year, und_price=p, r=r)
        call = CallOption(k, dte / steps_per_year, und_price=p, r=r)

        ivs[n, 0] = call.IV(g.call(k, dte))
        ivs[n, 1] = put.IV(g.put(k, dte))
    
    stored_results = np.array([[0.2396918,  0.24067023],
                               [0.21926788, 0.21988484],
                               [0.20051117, 0.20097515],
                               [0.18481161, 0.18526926],
                               [0.17360438, 0.17425838]])

    np.testing.assert_allclose(ivs, stored_results, atol=0.0001)
    plt.figure()
    moneyness = np.log(p / np.array(K))
    moneyness = K
    plt.plot(moneyness, ivs[:, 0], 'o', label='calls')
    plt.plot(moneyness, ivs[:, 1], 'x', label='puts')
    plt.legend()
    # plt.show()
    plt.savefig("egarch_skew.png")


def test_earnings():
    iv = 0.2
    p = 102.1
    dte = 50
    K = [95, 100, 105]
    w = iv**2 / 252
    a, b = 0, 0
    r = 0.015
    print("r", r)

    params = {'omega': -0.5426700217935021, 'alpha': 0.9392726911222145, 'theta': -0.16669601863907108, 'lam': 0.3138823220264745}

    g = egarch.EGARCHMonteCarloEarnings(p, 0, K, dte, 
        .2**2 / 252, **params,
        earnings_move_std=0.4, earnings_days_from_now=[10, 20],
        r=r, num_sims=5, days_in_year=252)
    g.run()

    plt.plot(g._price_paths)
    # breakpoint()
    # plt.show()
    plt.savefig("earnings_garch.png")