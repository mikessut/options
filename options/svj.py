"""
dS = mu*S*dt + sqrt(v)*S*dZ1 + (exp(alpha+delta*eps) - 1)*S*dq
dv = -lam*(v-v_avg)*dt + eta*sqrt(v)*dZ2
"""
import numpy as np
from scipy.integrate import quad
from options.heston import C, D


def psi(u, alpha, delta, lamJ):
    return -lamJ * 1j * u * (np.exp(alpha + delta**2/2) - 1) \
           + lamJ * (np.exp(1j * u * alpha - u**2*delta**2/2) - 1)
           

def phi(u, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T):
    return np.exp(C(u, lam, rho, eta, 0, T) * v_avg + D(u, lam, rho, eta, 0, T) * v) * \
           np.exp(psi(u, alpha, delta, lamJ) * T)


def integrand(u, lam, eta, rho, v_avg, v, lamJ, alpha, delta, k, T):
    return np.real(np.exp(-1j * u * k) * phi(u - 1j/2, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T)) / (u**2 + 0.25)


def call(strike, und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T):
    k = np.log(strike / und_price)
    integral = quad(integrand, 0, np.inf, args=(lam, eta, rho, v_avg, v, lamJ, alpha, delta, k, T))
    return und_price - np.sqrt(und_price * strike) / np.pi * integral[0]


def skew_integrand(u, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T):
    return u * np.imag(phi(u - 1j/2, lam, eta, rho, v_avg, v, lamJ, alpha, delta, T)) / (u**2 + 0.25)


def skew(lam, eta, rho, v_avg, v, lamJ, alpha, delta, T):
    """
    Eqn 5.8 ??
    """
    integral = quad(skew_integrand, 0, np.inf, args=(lam, eta, rho, v_avg, v, lamJ, alpha, delta, T))
    return -np.exp(v * T / 8) * np.sqrt(2 / np.pi) / np.sqrt(T) * integral[0]