"""
Jim Gatheral pg 63
"""
from options.merton import call
from options import CallOption
import pytest
import matplotlib.pyplot as plt
import numpy as np


@pytest.mark.parametrize(
    'sigma, lam, alpha, delta, expected',
    [
        (.2, .5, -.15, .05,  .2225),
        (.2, 1.0, -.07,  0,  .215),
        (.2, 1.0, -.07, .05, .2175),
    ]
)
def test_merton(sigma, lam, alpha, delta, expected):
    # sigma = .2
    # lam = .5
    # alpha = -.15
    # delta = .05
    T = 0.25
    strike = 100
    und_price = 100

    call_price = call(strike, lam, alpha, delta, sigma, T, und_price)
    c = CallOption(strike, T, und_bid=und_price, und_ask=und_price, r=0,
                   bid=call_price, ask=call_price)
    
    #assert pytest.approx(c.IV(), abs=.005) == .2225
    assert pytest.approx(c.IV(), abs=.004) == expected


def plt_fig52():
    k = np.linspace(-.5, .5, 100)
    und_price = 100
    strikes = np.exp(k) * und_price
    T = 0.25
    params = [
        (.2, .5, -.15, .05),
        (.2, 1.0, -.07,  0),
        (.2, 1.0, -.07, .05),
    ]
    for sigma, lam, alpha, delta in params:
        bsvol = []
        for strike in strikes:
            call_price = call(strike, lam, alpha, delta, sigma, T, und_price)
            c = CallOption(strike, T, und_bid=und_price, und_ask=und_price, r=0,
                           bid=call_price, ask=call_price)
            bsvol.append(c.IV())
        plt.plot(k, bsvol)
    plt.show()


#def plt_fig53():