import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar


class Option:

    def __init__(self, strike: float,
                 expiry: datetime.datetime = np.nan,
                 vol: float = np.nan,
                 und_price: float = np.nan,
                 r: float = .015,
                 bid: float = np.nan,
                 ask: float = np.nan):
        """
        :param strike: Option strike
        :param expiry: Either a datetime object of the expiry or a float
        :param vol: Volatility
        :param und_price: Price of underlying used for pricing calculations
        :param r: Risk free interest rate
        :param bid: Option bid price
        :param ask: Option ask price
        """
        self._strike = strike
        self._expiry = expiry
        self._vol = vol
        self._und_price = und_price
        self._r = r

        self._bid = bid
        self._ask = ask

    def set_bid(self, val):
        self._bid = val

    def set_ask(self, val):
        self._ask = val

    def bid(self) -> float:
        return self._bid

    def ask(self) -> float:
        return self._ask

    def mid(self) -> float:
        return (self.bid() + self.ask()) / 2

    def und_price(self) -> float:
        return self._und_price

    def set_und_price(self, val):
        self._und_price = val

    def extrinsic_val(self, price_func='mid', **kwargs) -> float:
        return getattr(self, price_func)(**kwargs) - self.intrinsic_val()

    def t_expiry(self) -> float:
        """
        Return time to expiry in years.
        """
        if isinstance(self._expiry, datetime.datetime):
            return (self._expiry - datetime.datetime.utcnow()).total_seconds() / 3600 / 24 / 365.25
        else:
            return self._expiry

    def BScalc(self, vol, t, r):
        d1 = 1 / vol / np.sqrt(t) * (np.log(self._und_price /
                                            self._strike) + (r + vol**2 / 2) * t)
        d2 = d1 - vol * np.sqrt(t)
        return d1, d2

    def IV(self, price, t=None, r=None):
        sol = root_scalar(lambda vol: self.BSprice(vol=vol, r=r, t=t) - price,
                          method='bisect',
                          bracket=(.0001, 3))
        return sol.root

    def gamma(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return norm.pdf(d1) / self._und_price / vol / np.sqrt(t)

    def vega(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return self._und_price * norm.pdf(d1) * np.sqrt(t)


class PutOption(Option):

    def BSprice(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return norm.cdf(-d2) * self._strike * np.exp(-r * t) - norm.cdf(-d1) * self._und_price

    def intrinsic_val(self) -> float:
        if self._strike > self._und_price:
            return self._strike - self._und_price
        else:
            return 0.0

    def delta(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return -norm.cdf(-d1)

    def theta(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return -self._und_price * norm.pdf(d1) * vol / 2 / np.sqrt(t) + r * self._strike * np.exp(-r * t) * norm.cdf(-d2)


class CallOption(Option):

    def BSprice(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return norm.cdf(d1) * self._und_price - norm.cdf(d2) * self._strike * np.exp(-r * t)

    def intrinsic_val(self) -> float:
        if self._strike < self._und_price:
            return self._und_price - self._strike
        else:
            return 0.0

    def delta(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r

        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return norm.cdf(d1)

    def theta(self, vol=None, t=None, r=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        d1, d2 = self.BScalc(vol, t, r)
        return -self._und_price * norm.pdf(d1) * vol / 2 / np.sqrt(t) - r * self._strike * np.exp(-r * t) * norm.cdf(d2)
