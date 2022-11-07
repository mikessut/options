"""
Conditional Heteroskedasticity in Asset Returns: A New Approach Nelson 1991
https://www.jstor.org/stable/2938260

arch package documentation is useful. That's where the sqrt(2 / np.pi) factor
came from, which is due to:
    Expectation(|zt|) = np.sqrt(2 / pi) 
    if zt is normal
 
E.g. np.mean(np.abs(np.random.randn(10000))) ~= np.sqrt(2 / pi)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from options.garch import MonteCarloOptionPricerBase
from options import cython_rec

_USE_CYTHON = True

class EGARCHMonteCarlo(MonteCarloOptionPricerBase):

    def __init__(self, p0, lr0, strikes, tdte, var0, 
        omega, alpha, theta, lam,
        r=0.015, mu=0, num_sims=5000,
        days_in_year=365.25):
        super().__init__(p0, strikes, tdte,
                         r, mu, num_sims, days_in_year)
        self._var0 = var0
        self._omega = omega
        self._alpha = alpha
        self._theta = theta
        self._lam = lam
        self._lr0 = lr0

    def _create_lrs(self):
        if _USE_CYTHON:
            cython_rec.egarch_create_lrs(self._lrs, self._var0, self._lr0, self._num_sims,
                                        max(self._days_to_expiration), 
                                        self._omega, self._alpha, self._theta, self._lam)
        else:
            for n in range(self._num_sims):
                sig2 = self._var0
                lsig2 = np.log(sig2)
                sig = np.sqrt(np.exp(lsig2))
                lr = self._lr0
                r = np.random.randn(max(self._days_to_expiration))
                for nd in range(max(self._days_to_expiration)):
                    lsig2 = self._omega + self._g(lr / sig, self._theta, self._lam) + self._alpha * lsig2
                    sig = np.sqrt(np.exp(lsig2))
                    sr = self._mu + sig * r[nd]
                    lr = np.log(1 + sr)
                    
                    self._lrs[nd, n] = lr

    @staticmethod
    def _g(e, theta, lam):
        return theta * e + lam * (np.abs(e) - np.sqrt(2 / np.pi))

    @staticmethod
    def calc_garch(lr, omega, alpha, theta, lam):
        lsig2 = np.zeros((len(lr), )) * np.nan
        lsig2[0] = omega / (1 - alpha)

        for n in range(1, len(lr)):
            sig_prev = np.sqrt(np.exp(lsig2[n-1]))
            lsig2[n] = omega + EGARCHMonteCarlo._g(lr[n-1] / sig_prev, theta, lam) + alpha * lsig2[n-1]

        return np.exp(lsig2) # returns sig2 = variance

    @staticmethod
    def _neg_logl(omega, alpha, theta, lam, lr):
        sig2 = EGARCHMonteCarlo.calc_garch(lr, omega, alpha, theta, lam)
        # return - (-np.log(sig2) - lr**2 / sig2).sum()
        return -norm(0, np.sqrt(sig2)).logpdf(lr).sum()  # Yes, this seems to give similar results!!!

    @staticmethod
    def fit(lr):
        X = minimize(lambda X, xx: EGARCHMonteCarlo._neg_logl(*X, xx), 
                    [np.log(.2**2 / 252) * (1 - 0.8), 0.8, -0.2, 0.1],
                    args=(lr, ),
                    method = 'Nelder-Mead',
                    options={'disp': True, 'maxiter': 1000})
        return {k: v for k, v in zip(['omega', 'alpha', 'theta', 'lam'], X.x)}

    def long_term_vol(*args, omega=None, alpha=None, theta=None, lam=None):
        """
        The shenanigans here are to make this both a static or instance method.
        """
        if len(args) == 1:
            self = args[0]
            return np.sqrt(np.exp(self._omega / (1 - self._alpha)) * 252)
        else:
            return np.sqrt(np.exp(omega / (1 - alpha)) * 252)


class EGARCHMonteCarloEarnings(EGARCHMonteCarlo):

    def __init__(self,
            p0, lr0, strikes, tdte, var0, 
            omega, alpha, theta, lam,
            earnings_move_std: float, earnings_days_from_now: list[int],
            r=0.015, mu=0, num_sims=5000, days_in_year=365.25
        ):
        super().__init__(p0, lr0, strikes, tdte, var0,
            omega, alpha, theta, lam,
            r, mu=mu, num_sims=num_sims, days_in_year=days_in_year)
        self._earnings_move_std = earnings_move_std
        self._earnings_days_from_now = earnings_days_from_now

    def _create_lrs(self):
        super()._create_lrs()
        # Now go in and add earnings moves
        for n in range(self._num_sims):
            for day in self._earnings_days_from_now:
                self._lrs[day-1, n] += np.random.randn() * self._earnings_move_std

