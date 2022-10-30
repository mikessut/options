from options import garch, portfolio
from options import PutOption, CallOption
import pytest
import pytz
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


def test_garch():
    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.00272673

    ndays = 53
    strikes = [500, 750, 1000, 1250, 1500, 2000, 3000, 4000]

    g = garch.GARCHMonteCarlo(1150, strikes, ndays, var0, w, alpha, beta)
    g.run()

    # let's look at 2000 strike with bid/ask of 10/18.  Model has price of 24,
    # suggesting it's profitable to be long.
    basis = 18
    strike = 2000
    # This is interesting that even though it is profitable in the long run, it
    # is only profitable ~5% of the time.
    assert g.call_pop(basis, strike, ndays) == pytest.approx(0.055, abs=.005)
    return g


def test_garch2():
    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.002365676345453833

    ndays = 25
    strikes = [1250]

    g = garch.GARCHMonteCarlo(1152, strikes, ndays, var0, w, alpha, beta)
    g.run()

    # let's look at 2000 strike with bid/ask of 10/18.  Model has price of 24,
    # suggesting it's profitable to be long.
    call_basis = 55  
    put_basis = 150 # Model value is ~181
    strike = strikes[0]
    # This is interesting that even though it is profitable in the long run, it
    # is only profitable ~5% of the time.
    # assert g.call_pop(basis, strike, ndays) == pytest.approx(0.055, abs=.005)
    expiry =  datetime.datetime(2022, 7, 29, 20, tzinfo=pytz.utc)
    now = expiry - datetime.timedelta(days=ndays)
    put = PutOption(strike, expiry, multiplier=0.1)
    call = CallOption(strike, expiry, multiplier=0.1)
    put._now = now
    call._now = now

    put_pos = portfolio.OptionPosition(put, 1, put_basis)
    call_pos = portfolio.OptionPosition(call, 1, call_basis)
    print(f"Call model value: {g.call(strike, ndays)}")
    print(f"Call POP: {g.call_pop(call_basis, strike, ndays):.2f} {g.pop(call_pos):.2f}")

    # How much could we increase POP if there was an offsetting delta also
    # underpriced?
    print(f"Put model value: {g.put(strike, ndays)}")
    print(f"Put POP: {g.put_pop(put_basis, strike, ndays):.2f} {g.pop(put_pos):.2f}")

    prt = portfolio.Portfolio([call_pos, put_pos])
    print(f"POP for put+call portfolio: {g.pop(prt):.2f}")
    return g


def test_garch3():
    """
    2022-07-05 22:21:50,935:__main__:I:<LXCallOptionContract 1250.0 0.065 Und: 1141.15 Bid/ask: 40.0 52.0>, 78.13, 26.13 -38.13 iv: 0.74 delta: 0.35 roi_long: 6.22 roi_short: -0.54
    2022-07-05 22:21:50,962:__main__:I:<LXPutOptionContract 1250.0 0.065 Und: 1141.15 Bid/ask: 200.0 369.8>, 182.83, -186.97 17.17 iv: 1.88 delta: -0.48 roi_long: -10.76 roi_short: 0.25
    """
    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.930**2 / 365
    und_price = 1141.15

    ndays = 24
    strikes = [1250]

    g = garch.GARCHMonteCarlo(und_price, strikes, ndays, var0, w, alpha, beta)
    g.run()

    # let's look at 2000 strike with bid/ask of 10/18.  Model has price of 24,
    # suggesting it's profitable to be long.
    call_basis = 52  
    put_basis = 200 # Model value is ~181
    strike = strikes[0]
    # This is interesting that even though it is profitable in the long run, it
    # is only profitable ~5% of the time.
    # assert g.call_pop(basis, strike, ndays) == pytest.approx(0.055, abs=.005)
    expiry =  datetime.datetime(2022, 7, 29, 20, tzinfo=pytz.utc)
    now = expiry - datetime.timedelta(days=ndays)
    put = PutOption(strike, expiry, multiplier=0.1)
    call = CallOption(strike, expiry, multiplier=0.1)
    put._now = now
    call._now = now

    put_pos = portfolio.OptionPosition(put, -1, put_basis)
    call_pos = portfolio.OptionPosition(call, 1, call_basis)
    print(f"Call model value: {g.call(strike, ndays)}")
    print(f"Call POP: {g.call_pop(call_basis, strike, ndays):.2f} {g.pop(call_pos):.2f}")

    # How much could we increase POP if there was an offsetting delta also
    # underpriced?
    print(f"Put model value: {g.put(strike, ndays)}")
    print(f"Put POP: {1-g.put_pop(put_basis, strike, ndays):.2f} {g.pop(put_pos):.2f}")

    prt = portfolio.Portfolio([call_pos, put_pos])
    print(f"Portfolio model value: {g.portfolio_expected_value(prt)}")
    print(f"POP for put+call portfolio: {g.pop(prt):.2f}")
    prt.pnl(1000, 1200)
    plt.savefig("pnl.png")
    return g


def test_pct_garch():
    strikes = [750, 1000, 1500, 2000]
    ndays = [22, 50]

    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.930**2 / 365
    und_price = 1141.15
    nsims = 16000

    call_prices1, put_prices1 = garch.GARCHMonteCarlo.garch_monte_carlo(und_price, strikes, ndays,
                          var0, w, alpha, beta, mu=0, num_sims=nsims, return_avgs=True)

    print(call_prices1.round(1))
    print(put_prices1.round(1))
    g = garch.GARCHMonteCarlo(und_price, strikes, ndays, var0, w, alpha, beta, num_sims=nsims)
    g.run()

    print()
    call_prices2 = np.zeros((len(strikes), len(ndays)))
    put_prices2 = np.zeros((len(strikes), len(ndays)))
    for i, strike in enumerate(strikes):
        for j, nday in enumerate(ndays):
            call_prices2[i, j] = g.call(strike, nday)
            put_prices2[i, j] = g.put(strike, nday)

    print("method 2")
    print(call_prices2.round(1))
    print(put_prices2.round(1))

    print()
    print((call_prices1 - call_prices2).round(1))
    print((put_prices1 - put_prices2).round(1))

    np.testing.assert_allclose(call_prices1, call_prices2, rtol=.05, atol=.5)
    np.testing.assert_allclose(put_prices1, put_prices2, rtol=.05, atol=.5)
    return g


def test_garch_new_day():
    und_price = 1000
    strikes = 1000
    ndays = [10, 20]

    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.930**2 / 365
    g = garch.GARCHMonteCarlo(und_price, strikes, ndays, var0, w, alpha, beta, num_sims=10000)
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


def test_garch_btc():
    und_price = 20829.7
    var0 = 0.0029350105294403744
    w = 0.00026337028025903464
    alpha = 0.13684325341862802
    beta = 0.7818352664119537

    g = garch.GARCHMonteCarlo(und_price, 21000, 7, var0, w, alpha, beta, num_sims=10000)
    g.run()

    print(g.call(21000, 7))


def test_fit_garch():
    btc = pd.read_pickle('tests/BTC.pd')
    eth = pd.read_pickle('tests/ETH.pd')

    btc['lr'] = np.log(btc.Close / btc.Close.shift())
    eth['lr'] = np.log(eth.Close / eth.Close.shift())

    print(f"hiv btc: {btc.lr[1:].std() * np.sqrt(365.25)} eth: {eth.lr[1:].std() * np.sqrt(365.25)} ")

    btc_fit = garch.fit_garch2(btc.Close.to_numpy())
    eth_fit = garch.fit_garch2(eth.Close.to_numpy())
    print(btc_fit, np.sqrt((btc_fit[0] / (1 - btc_fit[1] - btc_fit[2]) * 365.25)))
    print(eth_fit, np.sqrt((eth_fit[0] / (1 - eth_fit[1] - eth_fit[2]) * 365.25)))


def test_vs_bs():
    """
    Test against Black Scholles use alpha and beta = 0
    """
    # logging.basicConfig(level=logging.DEBUG)
    iv = 0.2
    p = 102.1
    dte = 50
    K = [95, 100, 105]
    w = iv**2 / 365.25
    a, b = 0, 0
    r = 0.05
    print("r", r)

    g = garch.GARCHMonteCarlo(p, K, dte, w, w, a, b, r=r, num_sims=200000, days_in_year=365.25)
    g.run()

    for k in K:
        put = PutOption(k, dte / 365.25, iv, und_price=p, r=r)
        call = CallOption(k, dte / 365.25, iv, und_price=p, r=r)

        print(f"put:  BS / MonteCarlo: {put.BSprice():.3f} {g.put(k, dte):.3f} {g.put(k, dte) - put.BSprice():.3f}")
        print(f"call: BS / MonteCarlo: {call.BSprice():.3f} {g.call(k, dte):.3f} {g.call(k, dte) - call.BSprice():.3f}")
        # assert pytest.approx(p.BSprice(), g.put(k, dte), abs=.01)
        # assert pytest.approx(c.BSprice(), g.call(k, dte), abs=.01)