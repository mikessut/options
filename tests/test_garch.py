from options import garch, portfolio
from options import PutOption, CallOption
import pytest
import pytz
import datetime
import matplotlib.pyplot as plt
import numpy as np


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
    nsims = 160000
    #nsims = 5000

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
            call_prices2[i, j] = g.call(strike, nday)[0]
            put_prices2[i, j] = g.put(strike, nday)[0]

    print(call_prices2.round(1))
    print(put_prices2.round(1))

    print()
    print((call_prices1 - call_prices2).round(1))
    print((put_prices1 - put_prices2).round(1))

    np.testing.assert_allclose(call_prices1, call_prices2, rtol=.05, atol=.5)
    np.testing.assert_allclose(put_prices1, put_prices2, rtol=.05, atol=.5)
    return g