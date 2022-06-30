"""
Price an option based off of all of the possible combinations of consecutive
days in a historical series.
"""
import numpy as np
from scipy.optimize import minimize
import arch


def put_price(p0, strike, historical_log_returns, days_to_expiration):
    """
    :param p0: Current price

    This is nice from it's simplicity, but lacks the real worldness of
    volatility clustering.  Would be nice to use something like GARCH
    """
    end_prices = []
    start_ctr = 0
    end_ctr = days_to_expiration

    while end_ctr <= len(historical_log_returns):
        end_prices.append(p0 * np.exp(historical_log_returns[start_ctr:end_ctr].sum()))
        end_ctr += 1
        start_ctr += 1

    put_prices = [0 if x > strike else strike - x for x in end_prices]
    return np.mean(put_prices)


def calc_garch(w, alpha, beta, var0, lr):
    var = np.zeros((len(lr),))
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
    return res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']


def garch_monte_carlo(p0, strikes, days_to_expiration, var0, w, alpha, beta, mu=0, num_sims=1000):

    strikes = np.array(strikes, ndmin=1)
    put_prices = np.zeros((len(strikes), num_sims))
    call_prices = np.zeros((len(strikes), num_sims))
    for n in range(num_sims):
        p_prev = p0
        var = var0
        for _ in range(days_to_expiration):
            p = p_prev * (1 + mu + np.sqrt(var) * np.random.randn())
            sr = 1 - p / p_prev
            var = w + alpha * sr**2 + beta * var
            p_prev = p
        for m, strike in enumerate(strikes):
            put_price = 0 if p > strike else strike - p
            call_price = 0 if p < strike else p - strike
            put_prices[m, n] = put_price
            call_prices[m, n] = call_price
    return np.mean(call_prices, axis=1), np.mean(put_prices, axis=1)
