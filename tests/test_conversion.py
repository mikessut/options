"""
http://www.optiontradingpedia.com/conversion_reversal_arbitrage.htm
"""
from options import PutOption, CallOption
from options.conversion import ReverseConversion, Conversion


def test_conversion():
  und_price = 51
  strike = 51
  c = CallOption(strike, bid=2.5, ask=2.5)
  p = PutOption(strike, bid=1.50, ask=1.50)

  conv = Conversion(p, c)
  print(conv.value())
  assert conv.value() == 1.0


def test_reverse_conversion():
  und_price = 51
  strike = 51
  c = CallOption(strike, bid=1.5, ask=1.5)
  p = PutOption(strike, bid=2.50, ask=2.50)

  conv = ReverseConversion(p, c)
  print(conv.value())
  assert conv.value() == 1.0