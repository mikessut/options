import arch
import numpy as np
from scipy.optimize import minimize


def calc_garch(w, alpha, beta, lr, var0=None):
    var = np.zeros((len(lr),))
    if var0 is None:
        # https://stats.stackexchange.com/questions/367722/fitting-a-garch1-1-model
        var[0] = w / (1 - alpha - beta)
    else:
        var[0] = var0
    for n in range(1, len(lr)):
        var[n] = w + alpha * lr[n-1]**2 + beta * var[n-1]
    var[var < 1e-12] = 1e-12
    return var


def fit_garch(historical_price, var0):
    """
    https://web-static.stern.nyu.edu/rengle/GARCH101.PDF
    https://www.youtube.com/watch?v=Xef4LL2KUy0

    r(t) = m + sqrt(h) * err
    r: return
    m: long term average
    h: variance of return
    err: standard normal distribution

    GARCH(1,1) Model
    sigma_n^2 = w + alpha * lr_(n-1)^2 + beta * sigma_(n-1)^2
    sigma^2 = var(iance)

    Use maximum likelihood approach, which is to adjust parameters to maximize likelihood.

    Likelihood defined as: -log(sigma^2) - ret^2 / sigma^2
    """

    lr = np.log(historical_price[1:] / historical_price[:-1])

    lr -= lr.mean()

    def neg_likelihood(X, var0, lr):
        w, alpha, beta = X
        w = w / 100
        #alpha = alpha / 10
        var = calc_garch(w, alpha, beta, var0, lr)
        return -np.sum(-np.log(var) - (lr)**2 / var)

    returns_sf = 100

    X = minimize(neg_likelihood, (0.1*100, .05, .8), 
                args=(var0, lr * returns_sf),
                #bounds=((0.0001, None), (0.0001, None), (0.0001, None)),
                method='SLSQP',
                options={'disp': True})
    return X.x[0] / returns_sf**2/100, X.x[1], X.x[2]   # w, alpha, beta


def fit_garch2(historical_price):
    """
    """
    # lr = np.log(historical_price[1:] / historical_price[:-1])
    sr = 1 - historical_price[1:] / historical_price[:-1]
    # Apparently this package want return in % so multiply by 100
    garch = arch.arch_model(sr, vol='garch', p=1, o=0, q=1, rescale=True)
    res = garch.fit()
    return res.params['omega'] / res.scale**2, res.params['alpha[1]'], res.params['beta[1]']


class GARCHMonteCarlo:

    def __init__(self, p0, strikes, days_to_expiration, var0, w, alpha, beta, mu=0, num_sims=5000):
        self._p0 = p0
        self._strikes = np.array(strikes, ndmin=1)
        self._days_to_expiration = np.array(days_to_expiration, ndmin=1, dtype=int)
        self._put_prices = np.zeros((len(strikes), len(days_to_expiration)))
        self._call_prices = np.zeros((len(strikes), len(days_to_expiration)))

        self._var0 = var0
        self._w = w
        self._alpha = alpha
        self._beta = beta
        self._mu = mu
        self._num_sims = num_sims

    def run(self):
        self._call_prices, self._put_prices = self.garch_monte_carlo(self._p0,
                                                                     self._strikes,
                                                                     self._days_to_expiration,
                                                                     self._var0,
                                                                     self._w,
                                                                     self._alpha,
                                                                     self._beta,
                                                                     self._mu,
                                                                     self._num_sims)

    def put(self, strike, days):
        idx_strike = np.where(self._strikes == strike)[0][0]
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        return self._put_prices[idx_strike, idx_days]

    def call(self, strike, days):
        idx_strike = np.where(self._strikes == strike)[0][0]
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        return self._call_prices[idx_strike, idx_days]

    @staticmethod
    def garch_monte_carlo(p0, strikes, days_to_expiration, var0, w, alpha, beta, mu=0, num_sims=1000):

        strikes = np.array(strikes, ndmin=1)
        days_to_expiration = np.array(days_to_expiration, ndmin=1, dtype=int)
        put_prices = np.zeros((len(strikes), len(days_to_expiration), num_sims))
        call_prices = np.zeros((len(strikes), len(days_to_expiration), num_sims))
        for n in range(num_sims):
            p_prev = p0
            var = var0
            for nd in range(max(days_to_expiration)):
                p = p_prev * (1 + mu + np.sqrt(var) * np.random.randn())
                sr = 1 - p / p_prev
                var = w + alpha * sr**2 + beta * var
                p_prev = p
                if (nd+1) in days_to_expiration:
                    idx = np.where(nd+1 == days_to_expiration)[0]
                    for m, strike in enumerate(strikes):
                        put_price = 0 if p > strike else strike - p
                        call_price = 0 if p < strike else p - strike
                        put_prices[m,idx, n] = put_price
                        call_prices[m, idx, n] = call_price
        return np.mean(call_prices, axis=2), np.mean(put_prices, axis=2)