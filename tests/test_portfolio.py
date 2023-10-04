from options.portfolio import *
from options import garch
from options import *
import pytest
import matplotlib.pyplot as plt


@pytest.mark.parametrize('num_shares', [1, 100, -1, -100])
def test_underlying(num_shares):
    u = UnderlyingPosition(num_shares, Quote(100, 100), basis=95)
    port = Portfolio([u])

    assert port.delta() == num_shares


def test_options():
    """
    Consider the following stylized example:  

    Current Price of Option1 (S)= $100Exercise Price of Option1 (X)= $100Risk
    FreeReturn (rf)=5% p.a.Time to Maturity (t)=91 days or 91/365 =
    24.93%Volatility ()= 20% p.a.

    he  resulting  Black-Scholes  call  and  put  prices  for  Option1  are
    $4.61  and  $3.37,  respectively

    https://www.economics-finance.org/jefe/volume11-2/04.delta%20gamma%20hedging%20and%20the%20black-scholes%20partial%20differential%20equation%20(1).pdf

    """

    S = 100
    K = 100
    r = .05
    q_und = Quote(S, S)
    c = CallOption(K, 91/365, .2, S, S, r)
    p = PutOption(K, 91/365, .2, S, S, r)

    assert pytest.approx(c.BSprice(), abs=.01) == 4.61
    assert pytest.approx(p.BSprice(), abs=.01) == 3.37

    assert pytest.approx(c.delta(), abs=0.0001) == 0.5694

    port = Portfolio([UnderlyingPosition(1000, q_und, 100), OptionPosition(c, -17.56, 0)])
    #assert pytest.approx(port.delta(), abs=.2) == 0
    return port


def plot_test_simple():
    S = 100
    K = 100
    r = .05
    q_und = Quote(S, S)
    c = CallOption(K, .001, .2, S, S, r)

    port = Portfolio([OptionPosition(c, 1)])

    p = np.linspace(90, 110, 100)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(p, port.sweep(port.value, p))
    ax[1].plot(p, port.sweep(port.delta, p))
    ax[2].plot(p, port.sweep(port.gamma, p))
    plt.show()


def plot_test():
    S = 100
    r = .05
    q_und = Quote(S, S)
    c = CallOption(105, .1, .2, S, S, r)
    p = PutOption(90, .1, .25, S, S, r)

    port = Portfolio([OptionPosition(c, -10),
                      OptionPosition(p, -1)])

    fig, ax = plt.subplots(3, 1, sharex=True)
    price = np.linspace(90, 110, 100)
    ax[0].plot(price, port.sweep(port.value, price))
    ax[1].plot(price, port.sweep(port.delta, price))
    ax[2].plot(price, port.sweep(port.gamma, price))

    d = port.delta(und_price=104)
    
    print(f"Delta: {d}")
    print(port)

    port.add_pos(UnderlyingPosition(-int(round(d)), q_und))

    d = port.delta(und_price=104)
    print(f"Delta: {d}")
    print(port)

    #fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(price, port.sweep(port.value, price))
    ax[1].plot(price, port.sweep(port.delta, price))
    ax[2].plot(price, port.sweep(port.gamma, price))
    plt.show()


def test_garch_portfolio():
    w, alpha, beta = (0.00026337028025903464, 0.13684325341862802, 0.7818352664119537)
    var0 = 0.930**2 / 365
    lr0 = .1

    und_price = 1255
    strikes = 1250
    ndays = 14

    g = garch.GARCHMonteCarlo(und_price, lr0, strikes, ndays, var0, w, alpha, beta, num_sims=10000)
    g.run()

    c = CallOption(strikes, 14/365.25)
    port = Portfolio([OptionPosition(c, -1, 0)])

    return g, c
