from options import garch, portfolio
from options import PutOption, CallOption
import pytest
import pytz
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import pathlib
import timeit


def test_garch():
    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.00272673

    ndays = 53
    strikes = [500, 750, 1000, 1250, 1500, 2000, 3000, 4000]

    g = garch.GARCHMonteCarlo(1150, 0, strikes, ndays, var0, w, alpha, beta)
    np.random.seed(1)
    g.run()

    # let's look at 2000 strike with bid/ask of 10/18.  Model has price of 24,
    # suggesting it's profitable to be long.
    basis = 18
    strike = 2000
    # This is interesting that even though it is profitable in the long run, it
    # is only profitable ~5% of the time.
    assert g.call_pop(basis, strike, ndays) == pytest.approx(0.055, abs=.005)
    return g


def test_garch_new_day():
    und_price = 1000
    strikes = 1000
    ndays = [10, 20]

    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.930**2 / 365
    g = garch.GARCHMonteCarlo(und_price, 0, strikes, ndays, var0, w, alpha, beta, num_sims=10000)
    g.run()

    calls = (g.call(strikes, 10), g.call(strikes, 20))
    puts = (g.put(strikes, 10), g.put(strikes, 20))
    print(calls)
    print(puts)

    # Making sure this doesn't fail
    print(g.call(strikes, 9))
    print(g.put(strikes, 9))

    print(g.call(strikes, 19))
    print(g.put(strikes, 19))

    # Make sure the above values are the same
    print(g.call(strikes, 10), g.call(strikes, 20))
    print(g.put(strikes, 10), g.put(strikes, 20))
    np.testing.assert_allclose(calls, (g.call(strikes, 10), g.call(strikes, 20)))
    np.testing.assert_allclose(puts, (g.put(strikes, 10), g.put(strikes, 20)))

    # assert False


@pytest.mark.parametrize('steps_per_year',
    [252, 365.25])
def test_vs_bs(steps_per_year):
    """
    Test against Black Scholles use alpha and beta = 0
    """
    # logging.basicConfig(level=logging.DEBUG)
    iv = 0.2
    p = 102.1
    dte = 50
    K = [90, 95, 100, 105, 110]
    w = iv**2 / steps_per_year
    a, b = 0, 0
    r = 0.015
    print("r", r)

    g = garch.GARCHMonteCarlo(p, 0, K, dte, w, w, a, b, r=r, num_sims=200000, days_in_year=steps_per_year)
    g.run()

    for p in [p, 103.4]:
        print("und price:", p)
        g.set_und_price(p)
        for n, k in enumerate(K):
            put = PutOption(k, dte / steps_per_year, iv, und_price=p, r=r)
            call = CallOption(k, dte / steps_per_year, iv, und_price=p, r=r)

            print(f"put:  BS / MonteCarlo: {put.BSprice():.3f} {g.put(k, dte):.3f} {g.put(k, dte) - put.BSprice():.3f}")
            print(f"call: BS / MonteCarlo: {call.BSprice():.3f} {g.call(k, dte):.3f} {g.call(k, dte) - call.BSprice():.3f}")
            assert pytest.approx(put.BSprice(), g.put(k, dte), abs=.02)
            assert pytest.approx(call.BSprice(), g.call(k, dte), abs=.02)


def test_garch_iv():
    df = pd.read_csv(pathlib.Path(__file__).parent / "SPY_20171104-20221104.csv")
    df.index = pd.DatetimeIndex(df.Date) 
    lr = np.log(df.Close / df.Close.shift())[1:]

    params = garch.GARCHMonteCarlo.fit(lr)
    print(f"Fit to historical data: {params}")

    p = 102.1
    dte = 50
    K = [90, 95, 100, 105, 110]
    steps_per_year = 252
    r = .015

    g = garch.GARCHMonteCarlo(p, -0.02, K, dte, .2**2 / 252, **params, r=r, num_sims=200000, days_in_year=steps_per_year)
    g.run()

    ivs = np.zeros((len(K), 2))

    for n, k in enumerate(K):
        put = PutOption(k, dte / steps_per_year, und_price=p, r=r)
        call = CallOption(k, dte / steps_per_year, und_price=p, r=r)

        ivs[n, 0] = call.IV(g.call(k, dte))
        ivs[n, 1] = put.IV(g.put(k, dte))
    
    plt.figure()
    moneyness = np.log(p / np.array(K))
    plt.plot(moneyness, ivs[:, 0], 'o', label='calls')
    plt.plot(moneyness, ivs[:, 1], 'x', label='puts')
    plt.legend()
    plt.savefig("garch_iv.png")


# @pytest.mark.skip("plots")
def test_earnings():
    iv = 0.2
    p = 102.1
    dte = 50
    K = [95, 100, 105]
    w = iv**2 / 252
    a, b = 0, 0
    r = 0.015
    print("r", r)

    g = garch.GARCHMonteCarloEarnings(p, 0, K, dte, w, w, a, b, 
        0.4, [10, 20],
        r=r, num_sims=5, days_in_year=252)
    g.run()

    plt.plot(g._price_paths)
    # breakpoint()
    # plt.show()
    plt.savefig("earnings_garch.png")


def benchmark_cython():
    df = pd.read_csv(pathlib.Path(__file__).parent / "SPY_20171104-20221104.csv")
    df.index = pd.DatetimeIndex(df.Date) 
    lr = np.log(df.Close / df.Close.shift())[1:]

    params = garch.GARCHMonteCarlo.fit(lr)
    print(f"Fit to historical data: {params}")

    p = 102.1
    dte = 50
    K = [90, 95, 100, 105, 110]
    steps_per_year = 252
    r = .015

    g = garch.GARCHMonteCarlo(p, -0.02, K, dte, .2**2 / 252, **params, r=r, num_sims=200000, days_in_year=steps_per_year)
    g_c = garch.GARCHMonteCarlo(p, -0.02, K, dte, .2**2 / 252, **params, r=r, num_sims=200000, days_in_year=steps_per_year)

    garch._USE_CYTHON = False
    print("python: ", timeit.timeit(g.run, number=1))
    garch._USE_CYTHON = True
    print("cython: ", timeit.timeit(g_c.run, number=1))

    print(g.call(90, 50), g_c.call(90, 50))