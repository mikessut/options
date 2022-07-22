from options import Option, Quote, PutOption, CallOption
from options import garch
from typing import List, Iterator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd


class Position:

    def __init__(self, qty: int, basis: float):
        """
        qty is negative for a short position
        basis is always positive????  It represents the per unit
        """
        self._qty = qty
        self._basis = basis

    def delta(self, opt_price=None, vol=None, t=None, r=None, und_price=None) -> float:
        raise NotImplemented()

    def gamma(self, opt_price=None, vol=None, t=None, r=None, und_price=None) -> float:
        raise NotImplemented()

    def value(self, opt_price=None, vol=None, t=None, r=None, und_price=None) -> float:
        raise NotImplemented()

    @property
    def basis(self) -> float:
        return self._basis

    @property
    def qty(self) -> int:
        return self._qty

    def cost(self) -> float:
        return self.qty * self.basis


class OptionPosition(Position):
    
    def __init__(self, opt: Option, qty: int, basis: float):
        super().__init__(qty, basis)
        self._opt = opt

    @property
    def option(self) -> Option:
        return self._opt

    def delta(self):
        return self.qty * self.option.multiplier * self.option.delta()

    def theta(self):
        return self.qty * self.option.multiplier * self.option.theta()

    def gamma(self):
        return self.qty * self.option.multiplier * self.option.gamma()

    def value(self, opt_price=None, vol=None, t=None, r=None, und_price=None):
        """
        If no arguments are passed, use bid/ask price to close position.

        If any arguments are passed, use BS pricing.
        """
        if all([x is None for x in [vol, t, r, und_price]]):
            if self._qty < 0:
                # Buy to close => ask
                return self._qty * self._opt.ask()
            else:
                # Sell to close => bid
                return self._qty * self._opt.bid()
        else:
            # Use BS pricing
            return self._qty * self._opt.BSprice(vol, t, r, und_price)

    def cost(self) -> float:
        return self.qty * self.basis * self.option.multiplier

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} of {self._opt} x {self._qty}>"


class UnderlyingPosition(Position):

    def __init__(self, qty, q: Quote, basis: float):
        super().__init__(qty, basis)
        self._quote = q

    def delta(self):
        return self._qty

    def gamma(self):
        return 0.0

    def theta(self):
        return 0.0

    def value(self, vol=None, t=None, r=None, und_price=None):
        if all([x is None for x in [vol, t, r, und_price]]):
            if self._qty < 0:
                # Buy to close => ask
                return self._qty * self._quote.ask()
            else:
                # Sell to close => bid
                return self._qty * self._quote.bid()
        else:
            # Use passed und_price
            return self._qty * und_price

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} qty: {self._qty}>"


class Portfolio:

    def __init__(self, positions: List[Position]):
        self._positions = positions

    def add_pos(self, pos: Position):
        self._positions.append(pos)

    def delta(self):
        return sum([x.delta() for x in self])

    def gamma(self):
        return sum([x.gamma() for x in self])

    def theta(self):
        """
        This is annualized.
        
        Negative number is an expense (e.g. long option)
        """
        return sum([x.theta() for x in self])

    def value(self):
        return sum([x.value() for x in self])

    def sweep(self, func, und_price):
        results = []
        for p in und_price:
            results.append(func(und_price=p))
        return results

    def __getitem__(self, n):
        return self._positions[n]

    def __iter__(self) -> Iterator[Position]:
        return iter(self._positions)

    def __repr__(self) -> str:
        s = f'<{self.__class__.__name__}\n'
        for p in self._positions:
            s += f'  {p}\n'
        s += '>'
        return s

    def table(self) -> str:
        cols = "pos,qty,expiry,model_price,delta,theta"
        result = cols + '\n'
        for p in self:
            if isinstance(p, OptionPosition):
                result += f"{p},{p.qty},{p.option.expiry.strftime('%Y%m%d')},{p.option.price},{p.delta()},{p.theta()}\n"
            else:
                result += f"{p},{p.qty},,,{p.delta()},{p.theta()}\n"
        return result

    def dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=['pos', 'qty', 'expiry', 'strike', 'model_price', 'model_vol', 'bid', 'ask', 'basis', 'delta', 'prt_delta', 'theta'])
        
        ctr = 0
        for p in self:
            df.loc[ctr, 'pos'] = p
            df.loc[ctr, 'qty'] = p.qty
            df.loc[ctr, 'prt_delta'] = p.delta()
            df.loc[ctr, 'theta'] = p.theta() / 365.25
            df.loc[ctr, 'basis'] = p.basis
            if isinstance(p, OptionPosition):
                df.loc[ctr, 'expiry'] = p.option.expiry.strftime('%Y%m%d')
                df.loc[ctr, 'strike'] = p.option.strike
                df.loc[ctr, 'model_price'] = p.option.model_price
                df.loc[ctr, 'model_vol'] = p.option.vol
                df.loc[ctr, 'bid'] = p.option.bid
                df.loc[ctr, 'ask'] = p.option.ask
                df.loc[ctr, 'delta'] = p.option.delta()
            ctr += 1
        return df

    def pnl(self, minp, maxp, exclude_und=False, garch_instance=None):
        """
        pnl plot at expiry of all positions
        """
        price = np.linspace(minp, maxp, 100)
        pnl = np.zeros(price.shape)

        if garch_instance is not None:
            pnl_today = np.zeros(price.shape)

        for pos in self:
            if isinstance(pos, OptionPosition):
                pnl -= pos.option.multiplier * pos.qty * pos.basis
                if isinstance(pos.option, CallOption):
                    idx = price > pos.option.strike
                    pnl[idx] += (price[idx] - pos.option.strike) * pos.option.multiplier * pos.qty
                    if garch_instance is not None:
                        for n in range(len(price)):
                            garch_instance.set_und_price(price[n])
                            pnl_today[n] += (garch_instance.call(pos.option.strike, int(np.round(pos.option.t_expiry() * 365.25)), pos.option.expiry) - pos.basis) * pos.option.multiplier * pos.qty
                elif isinstance(pos.option, PutOption):
                    idx = price < pos.option.strike
                    pnl[idx] += (pos.option.strike - price[idx]) * pos.option.multiplier * pos.qty
                    if garch_instance is not None:
                        for n in range(len(price)):
                            garch_instance.set_und_price(price[n])
                            pnl_today[n] += (garch_instance.put(pos.option.strike, int(np.round(pos.option.t_expiry() * 365.25)), pos.option.expiry) - pos.basis) * pos.option.multiplier * pos.qty
                else:
                    raise TypeError(f"unknown position type {pos.option}")
            else:
                # underlying pos
                if not exclude_und:
                    pnl += pos.qty * (price - pos.basis)

        plt.figure()
        plt.plot(price, pnl)
        if garch_instance is not None:
            plt.plot(price, pnl_today)
        plt.xlabel('Price')
        plt.ylabel('PnL')
        plt.title(datetime.datetime.utcnow().isoformat())

    def delta_plot(self, minp, maxp, exclude_und=False, garch_instance=None):
        """
        pnl plot at expiry of all positions
        """
        price = np.linspace(minp, maxp, 100)
        delta = np.zeros(price.shape)

        if garch_instance is not None:
            pnl_today = np.zeros(price.shape)

        for pos in self:
            if isinstance(pos, OptionPosition):
                pnl -= pos.option.multiplier * pos.qty * pos.basis
                if isinstance(pos.option, CallOption):
                    idx = price > pos.option.strike
                    pnl[idx] += (price[idx] - pos.option.strike) * pos.option.multiplier * pos.qty
                    if garch_instance is not None:
                        for n in range(len(price)):
                            garch.set_und_price(price[n])
                            pnl_today[n] += (garch_instance.call(pos.option.strike, int(np.round(pos.option.t_expiry() * 365.25))) - pos.basis) * pos.option.multiplier * pos.qty
                elif isinstance(pos.option, PutOption):
                    idx = price < pos.option.strike
                    pnl[idx] += (pos.option.strike - price[idx]) * pos.option.multiplier * pos.qty
                    if garch_instance is not None:
                        for n in range(len(price)):
                            garch.set_und_price(price[n])
                            pnl_today[n] += (garch_instance.put(pos.option.strike, int(np.round(pos.option.t_expiry() * 365.25))) - pos.basis) * pos.option.multiplier * pos.qty
                else:
                    raise TypeError(f"unknown position type {pos.option}")
            else:
                # underlying pos
                if not exclude_und:
                    pnl += pos.qty * (price - pos.basis)

        plt.figure()
        plt.plot(price, pnl)
        if garch:
            plt.plot(price, pnl_today)
        plt.xlabel('Price')
        plt.ylabel('delta')
        plt.title(datetime.datetime.utcnow().isoformat())
