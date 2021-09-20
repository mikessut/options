"""
dX = mu*S*dt + sigma*S*dZ + (exp(alpha+delta*eps) - 1)*S*dq
"""
import numpy as np
from scipy.integrate import quad


def w(alpha, delta, lam, sigma):
    return -0.5*sigma**2 - lam*(np.exp(alpha+delta**2/2) - 1)


def phi(u, alpha, delta, lam, sigma, T):
    return np.exp(1j * u * w(alpha, delta, lam, sigma) * T \
                  - .5 * u**2 * sigma**2 * T \
                  + lam * T * (np.exp(1j*u*alpha - u**2*delta**2/2) - 1))


def integrand(u, alpha, delta, lam, sigma, T, k):
    return np.real(np.exp(-1j * u * k) * phi(u - 1j/2, alpha, delta, lam, sigma, T)) / (u**2 + 0.25)


def call(strike, lam, alpha, delta, sigma, T, und_price):
    """
    Computed using eqn 5.6
    """
    k = np.log(strike / und_price)
    integral = quad(integrand, 0, np.inf, args=(alpha, delta, lam, sigma, T, k))
    return und_price - np.sqrt(und_price * strike) / np.pi * integral[0]


def skew_integrand(u, alpha, delta, lam, sigma, T):
    return u * np.imag(phi(u - 1j/2, alpha, delta, lam, sigma, T)) / (u**2 + 0.25)


def skew(lam, alpha, delta, sigma, T):
    integral = quad(skew_integrand, 0, np.inf, args=(alpha, delta, lam, sigma, T))
    return -np.sqrt(2 / np.pi) / np.sqrt(T) * integral[0]