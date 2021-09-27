import abc
from abc import abstractmethod
from options import Option, Quote
from typing import List


class Position(abc.ABC):

    @abstractmethod
    def delta(self, vol=None, t=None, r=None, und_price=None) -> float:
        pass

    @abstractmethod
    def gamma(self, vol=None, t=None, r=None, und_price=None) -> float:
        pass

    @abstractmethod
    def value(self, vol=None, t=None, r=None, und_price=None) -> float:
        pass


class OptionPosition(Position):
    
    def __init__(self, opt: Option, qty: int):
        self._opt = opt
        self._qty = qty

    def delta(self, vol=None, t=None, r=None, und_price=None):
        return self._qty * self._opt.delta(vol, t, r, und_price)

    def gamma(self, vol=None, t=None, r=None, und_price=None):
        return self._qty * self._opt.gamma(vol, t, r, und_price)

    def value(self, vol=None, t=None, r=None, und_price=None):
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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} of {self._opt} x {self._qty}>"


class UnderlyingPosition(Position):

    def __init__(self, qty, q: Quote):
        self._qty = qty
        self._quote = q

    def delta(self, vol=None, t=None, r=None, und_price=None):
        return self._qty

    def gamma(self, vol=None, t=None, r=None, und_price=None):
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

    def delta(self, und_price=None):
        return sum([x.delta(und_price=und_price) for x in self._positions])

    def gamma(self, und_price=None):
        return sum([x.gamma(und_price=und_price) for x in self._positions])

    def value(self, und_price=None):
        return sum([x.value(und_price=und_price) for x in self._positions])

    def sweep(self, func, und_price):
        results = []
        for p in und_price:
            results.append(func(und_price=p))
        return results

    def __repr__(self) -> str:
        s = '<{self.__class__.__name__}\n'
        for p in self._positions:
            s += f'  {p}\n'
        s += '>'
        return s
