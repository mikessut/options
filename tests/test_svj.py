from options import CallOption
import pytest
from options.svj import call, skew
import numpy as np
import matplotlib.pyplot as plt


def test_svj():
    lam = 2.03
    eta = .38
    rho = -.57
    v_avg = .04
    v = .04
    lamJ = .59
    alpha = -.05
    delta = .07

    strike = 100
    und_price = 100
    T = .25

    call_price = call(strike, und_price, lam, eta, rho,
                      v_avg, v, lamJ, alpha, delta, T)
    print(call_price)
    #assert False


def atm_skew():
    # BCC parameters
    lam = 2.03
    eta = .38
    rho = -.57
    v_avg = .04
    v = .04
    lamJ = .59
    alpha = -.05
    delta = .07

    strike = 100
    und_price = 100

    Ts = np.linspace(.01, 1, 20)

    skews = []
    for T in Ts:
        skews.append(skew(lam, eta, rho, v_avg, v, lamJ, alpha, delta, T))

    skews = np.array(skews)**2
    plt.plot(Ts, skews)


def table_55():
    v =         .0158
    v_avg =     .0439
    eta =       .3038
    rho =       -.6974
    lam =       .5394
    lamJ =      .1308
    delta =     .0967
    alpha =     -.1151
    und_price = 100

    T, k = np.meshgrid(np.linspace(.1, 1.5, 20),
                       np.linspace(-.5, .5, 20))
    Z = np.zeros(T.shape)

    for m in range(T.shape[0]):
        for n in range(T.shape[1]):
            strike = np.exp(k[m, n]) * und_price
            call_price = call(strike, und_price, lam, eta, rho,
                              v_avg, v, lamJ, alpha, delta, T[m, n])
            c = CallOption(strike, T[m, n], und_bid=und_price, und_ask=und_price, r=0,
                           bid=call_price, ask=call_price)
            Z[m, n] = c.IV()
        
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(T, k, Z)


@pytest.mark.parametrize(
    "strike, expected",
    [
        (0.8, .219081),
        (.9, .139729),
        (1.0, .075586),
        (1.1, .032256),
        (1.2, .010744),
    ]
)
def test_alex(strike, expected):
    """
    https://github.com/alexbadran/SVJ_Model/blob/master/Major%20Project.pdf
    pg 12
    """
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

    call_price = call(strike, und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T)
    assert pytest.approx(call_price, abs=.0001) == expected