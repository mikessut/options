import datetime
import pytz
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar


class Option:

    def __init__(self, strike: float,
                 expiry: datetime.datetime = np.nan,
                 vol: float = np.nan,
                 und_bid: float = np.nan,
                 und_ask: float = np.nan,
                 r: float = .015,
                 bid: float = np.nan,
                 ask: float = np.nan,
                 und_price=None,
                 opt_price=None,
                 multiplier=100,
                 metadata={}):
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
        self._und_bid = und_bid
        self._und_ask = und_ask
        self._r = r

        self._bid = bid
        self._ask = ask

        if und_price is not None:
            self._und_bid = und_price
            self._und_ask = und_price

        if opt_price is not None:
            self._bid = opt_price
            self._ask = opt_price

        self._multiplier = multiplier
        self._metadata = metadata

        self._model_price = np.nan

        self._now = None  # Probably only used in testing

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

    def set_und_bid(self, val):
        self._und_bid = val

    def set_und_ask(self, val):
        self._und_ask = val

    def set_und_price(self, val):
        self._und_bid = val
        self._und_ask = val

    def und_mid(self) -> float:
        return (self._und_bid + self._und_ask) / 2

    def und_bid(self) -> float:
        return self._und_bid

    def und_ask(self) -> float:
        return self._und_ask

    def set_model_price(self, val):
        self._model_price = val

    def model_price(self) -> float:
        return self._model_price

    def short_excess_value(self):
        return self.bid() - self.model_price()

    def long_excess_value(self):
        return self.model_price() - self.ask()

    def long_roi(self):
        return np.log(self.model_price() / self.ask()) / self.t_expiry()

    # def set_und_price(self, val):
    #     self._und_price = val

    def extrinsic_val(self, price_func='mid', und_price_func='und_mid') -> float:
        return getattr(self, price_func)() - self.intrinsic_val(und_price_func=und_price_func)

    @property
    def expiry(self) -> datetime.datetime:
        return self._expiry

    def t_expiry(self) -> float:
        """
        Return time to expiry in years.
        """
        if isinstance(self._expiry, datetime.datetime):
            if self._now is not None:
                return (self._expiry - self._now).total_seconds() / 3600 / 24 / 365.25
            else:
                return (self._expiry - pytz.utc.localize(datetime.datetime.utcnow())).total_seconds() / 3600 / 24 / 365.25
        else:
            return self._expiry

    @property
    def strike(self):
        return self._strike

    @property
    def multiplier(self):
        return self._multiplier

    def BScalc(self, vol, t, r, und_price):
        if not all([np.isfinite(x) for x in [vol, t, r, und_price]]):
            raise ValueError("All parameters must be finite")
        d1 = 1 / vol / np.sqrt(t) * (np.log(und_price /
                                            self._strike) + (r + vol**2 / 2) * t)
        d2 = d1 - vol * np.sqrt(t)
        return d1, d2

    def IV(self, price=None, t=None, r=None, und_price=None):
        if price is None:
            price = self.mid()
            if price < self.intrinsic_val():
                price = np.mean([self.intrinsic_val(), self.ask()])
        try:
            sol = root_scalar(lambda vol: self.BSprice(vol=vol, r=r, t=t, und_price=und_price) - price,
                            method='bisect',
                            bracket=(.0001, 3))
            return sol.root
        except ValueError as e:
            print("Warning:", e)
            return np.nan

    def gamma(self, vol=None, t=None, r=None, und_price=None):
        """
        Change in delta with respect to change in underlying price.
        """
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return norm.pdf(d1) / und_price / vol / np.sqrt(t)

    def vega(self, vol=None, t=None, r=None, und_price=None):
        """
        Change in option price with changes to volatility
        """
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return self.und_mid() * norm.pdf(d1) * np.sqrt(t)

    def std_moneyness(self):
        """
        https://en.wikipedia.org/wiki/Moneyness
        """
        if not np.isfinite(self._vol):
            vol = self.IV()
        else:
            vol = self._vol
        return (np.log(self.und_mid() / self._strike) + self._r * self.t_expiry()) / vol / np.sqrt(self.t_expiry())

    def moneyness(self):
        F = self.und_mid() * np.exp(self._r * self.t_expiry())
        return np.log(F / self._strike)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._strike} {self.t_expiry():.3f} Und: {(self.und_ask() + self.und_bid())/2:.2f} Bid/ask: {self.bid()} {self.ask()}>"


class PutOption(Option):

    def BSprice(self, vol=None, t=None, r=None, und_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return norm.cdf(-d2) * self._strike * np.exp(-r * t) - norm.cdf(-d1) * und_price

    def intrinsic_val(self, und_price_func='und_mid') -> float:
        if self._strike > getattr(self, und_price_func)():
            return self._strike - getattr(self, und_price_func)()
        else:
            return 0.0

    def delta(self, vol=None, t=None, r=None, und_price=None, opt_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if opt_price is not None:
            # Calculate the IV for this price
            if vol is not None:
                raise ValueError("Both vol and opt_price cannot be specified")
            vol = self.IV(opt_price, t, r, und_price)
        elif vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return -norm.cdf(-d1)

    def theta(self, vol=None, t=None, r=None, und_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return -und_price * norm.pdf(d1) * vol / 2 / np.sqrt(t) + r * self._strike * np.exp(-r * t) * norm.cdf(-d2)

    def short_roi(self):
        """
        Cash secured at strike price.
        """
        PV = self.strike - self.bid()
        FV = self.strike - self.model_price()
        return np.log(FV / PV) / self.t_expiry()


class CallOption(Option):

    def BSprice(self, vol=None, t=None, r=None, und_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return norm.cdf(d1) * und_price - norm.cdf(d2) * self._strike * np.exp(-r * t)

    def intrinsic_val(self, und_price_func='und_mid') -> float:
        if self._strike < getattr(self, und_price_func)():
            return getattr(self, und_price_func)() - self._strike
        else:
            return 0.0

    def delta(self, vol=None, t=None, r=None, und_price=None, opt_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if opt_price is not None:
            # Calculate the IV for this price
            if vol is not None:
                raise ValueError("Both vol and opt_price cannot be specified")
            vol = self.IV(opt_price, t, r, und_price)
        elif vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return norm.cdf(d1)

    def theta(self, vol=None, t=None, r=None, und_price=None):
        if t is None:
            t = self.t_expiry()
        if r is None:
            r = self._r
        if vol is None:
            vol = self._vol
        if und_price is None:
            und_price = self.und_mid()
        d1, d2 = self.BScalc(vol, t, r, und_price)
        return -und_price * norm.pdf(d1) * vol / 2 / np.sqrt(t) - r * self._strike * np.exp(-r * t) * norm.cdf(d2)

    def short_roi(self):
        """
        "Covered call" - secured by holding underlying at current undlying price.
        """
        PV = self.und_mid() - self.bid()
        FV = self.und_mid() - self.model_price()
        return np.log(FV / PV) / self.t_expiry()


class Quote:

    def __init__(self, bid, ask):
        self._bid = bid
        self._ask = ask

    def bid(self):
        return self._bid

    def ask(self):
        return self._ask


def put_parity(und_price, strike, call_price, r, T):
    return call_price - und_price + strike * np.exp(-r * T)


def call_parity(und_price, strike, put_price, r, T):
    return und_price - strike * np.exp(-r * T) + put_price
