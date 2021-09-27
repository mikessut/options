from options import random_walk
from options import svj
import pytest
import numpy as np


def plot_svj_walks():
    
    und_price = 1
    v = .04
    v_avg = .04
    T = 1
    rho = -.64
    eta = .39
    #eta = .2
    lam = 1.15
    alpha = -.1151
    delta = .0967
    lamJ = .1308
    
    mu = 0  # or risk free rate?

    svj = random_walk.SVJ(und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, mu, 1/365)
    svj.nsteps(10*365)
    ax = svj.plot()
    print(svj._num_jumps)

    for _ in range(10):
        svj = random_walk.SVJ(und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, mu, 1/365)
        svj.nsteps(10*365)
        svj.plot(ax)
        print(svj._num_jumps)


        
def test_monte_carlo_svj():
    strike, expected = 1.0, .075586
    
    und_price = 1
    v = .04
    v_avg = .04
    T = 1
    rho = -.64
    eta = .39
    lam = 1.15
    alpha = -.1151
    delta = .0967
    lamJ = .1308
    mu = 0
    dt = 1/365

    call_price = svj.call(strike, und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T)
    
    # first off all make sure expected is corred
    assert pytest.approx(call_price, abs=.0001) == expected
    
    final_call_price = []
    for _ in range(1000):
        svj_walk = random_walk.SVJ(und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, mu, dt)
        svj_walk.nsteps(int(T / dt))
        final_call_price.append(max(0, svj_walk._price[-1] - strike))
    print(expected, np.mean(final_call_price))
    assert expected == np.mean(final_call_price)
                                