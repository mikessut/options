"""
Price an option based off of all of the possible combinations of consecutive
days in a historical series.
"""
import numpy as np
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


