from options.heston import call
from options import CallOption
import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_heston_call():
    """
    https://kluge.in-chemnitz.de/tools/pricer/
    """

    tau = 0.5
    strike = 123.4
    r = 0
    und_price = 123.4
    v = 0.1197**2
    v_avg = 0.108977**2
    eta = 0.33147
    lam = 1.98937
    rho = 0.0258519

    assert pytest.approx(call(strike, lam, rho, eta, tau, v, v_avg, r, und_price), abs=.01) == 3.731032971

    strike = 130
    assert pytest.approx(call(strike, lam, rho, eta, tau, v, v_avg, r, und_price), abs=.01) == 1.540944


def plot_skew():

    tau = 0.1
    r = .015
    und_price = 3000
    v = .0174  #1.2**2
    v_avg = .0354  # 1**2
    eta = .4
    lam = 1.3
    rho = -.72

    strikes = np.linspace(2000, 5000, 15)  # [90, 95, 100, 105, 110]

    plt.figure()
    for tau in [.1, .5, 1]:
        m = []
        iv = []

        for strike in strikes:
            call_price = call(strike, lam, rho, eta, tau, v, v_avg, r, und_price)
            c = CallOption(strike, tau,
                            und_bid=und_price, und_ask=und_price, r=r,
                            bid=call_price, ask=call_price)
            iv.append(c.IV())
            m.append(c.moneyness())

        plt.plot(strikes, iv)

    plt.show()