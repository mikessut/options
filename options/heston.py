"""
dS = mu * S * dt + sqrt(v) * S * dZ1
dv = -lambda * (v - v_avg)*dt + eta*sqrt(v) * dZ2

Z1 and Z2 correlated by rho

Follows the derivation in Jim Gatheral's "The Volatility Surface" pg. 19.
"""
import numpy as np
from scipy.integrate import quad
import options


def alpha(u, j):
    return -u**2 / 2 - 1j * u / 2 + 1j * j * u


def beta(u, lam, rho, eta, j):
    return lam - rho * eta * j - rho * eta * 1j * u 


def gamma(eta):
    return eta**2 / 2


def d(u, lam, rho, eta, j):
    return np.sqrt(beta(u, lam, rho, eta, j)**2 - 4*alpha(u, j) * gamma(eta))


def r(plus: bool, u, lam, rho, eta, j):
    if plus:
        return (beta(u, lam, rho, eta, j) + d(u, lam, rho, eta, j)) / eta**2
    else:
        return (beta(u, lam, rho, eta, j) - d(u, lam, rho, eta, j)) / eta**2


def r_neg(u, lam, rho, eta, j):
    return r(False, u, lam, rho, eta, j)


def r_pos(u, lam, rho, eta, j):
    return r(True, u, lam, rho, eta, j)


def g(u, lam, rho, eta, j):
    return r_neg(u, lam, rho, eta, j) / r_pos(u, lam, rho, eta, j)


def D(u, lam, rho, eta, j, tau):
    expdt = np.exp(-d(u, lam, rho, eta, j) * tau)
    return r_neg(u, lam, rho, eta, j) * ((1 - expdt) / (1 - g(u, lam, rho, eta, j) * expdt))


def C(u, lam, rho, eta, j, tau):
    gfunc = g(u, lam, rho, eta, j)
    logf = np.log((1 - gfunc * np.exp(-d(u, lam, rho, eta, j) * tau)) / (1 - gfunc))
    return lam * (r_neg(u, lam, rho, eta, j) * tau - 2/eta**2 * logf)


def integrand(u, lam, rho, eta, j, tau, v, v_avg, x):
    return np.real(np.exp(C(u, lam, rho, eta, j, tau) * v_avg + D(u, lam, rho, eta, j, tau) * v + 1j * u * x) / (1j * u))


def P(x, lam, rho, eta, j, tau, v, v_avg):
    """
    :param j: Index either 0 or 1
    """
    integral = quad(integrand, 0, np.inf, args=(lam, rho, eta, j, tau, v, v_avg, x))
    # print(integral)
    return 0.5 + 1 / np.pi * integral[0]


def call(strike, lam, rho, eta, tau, v, v_avg, r, und_price):
    F = und_price * np.exp(r * tau)
    x = np.log(F / strike)
    return strike * (np.exp(x) * P(x, lam, rho, eta, 1, tau, v, v_avg) - P(x, lam, rho, eta, 0, tau, v, v_avg))


def put(strike, lam, rho, eta, tau, v, v_avg, r, und_price):
    call_price = call(strike, lam, rho, eta, tau, v, v_avg, r, und_price)
    return options.put_parity(und_price, strike, call_price, r, tau)
