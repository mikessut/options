from options import PutOption, CallOption
import numpy as np


class Conversion:
    """
    Long underlying
    Long put
    Short call
    """

    def __init__(self, put: PutOption, call: CallOption):
        self._put = put
        self._call = call
        if np.isfinite(put.und_mid()):
            assert put.und_ask() == call.und_ask(), f"Put und_ask: {put.und_ask()}; call und_bid: {call.und_ask()}"
            assert put.und_bid() == call.und_bid(), f"Put und_bid: {put.und_bid()}; call und_bid: {call.und_bid()}"

    def value(self, price_func='taker'):
        """
        Short extrinsic value minus long intrinsic value
        """

        return self._call.extrinsic_val('bid', und_price_func='und_ask') - \
               self._put.extrinsic_val('ask', und_price_func='und_ask')

    def cost(self):
        """
        Cost to put on position
        """
        return self._put.und_ask() \
               + self._put.ask() \
               - self._call.bid()

    def value_pct(self):
        return self.value() / self.cost()


class ReverseConversion:
    """
    Short underlying
    Long Call
    Short put
    """

    def __init__(self, put: PutOption, call: CallOption):
        self._put = put
        self._call = call

    def value(self, price_func='taker'):
        """
        Short extrinsic value minus long intrinsic value
        """
        return self._put.extrinsic_val('bid', und_price_func='und_bid') - \
               self._call.extrinsic_val('ask', und_price_func='und_bid')

    def cost(self):
        """
        Cost to put on position
        """
        return -self._put.und_bid() \
               + self._call.ask() \
               - self._put.bid()

    def value_pct(self):
        return self.value() / self.cost()