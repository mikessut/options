"""
Good calculator to check:
https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
"""
import options
import datetime
import numpy as np
import pytest


def test_BS_value():
  c = options.CallOption(110, expiry=0.5, vol=.5, und_bid=100, und_ask=100)
  p = options.PutOption(110, expiry=0.5, vol=.5, und_bid=100, und_ask=100)
  assert c.BSprice() == 10.531783818155567
  assert p.BSprice() == 19.709869848260794
  c.price = c.BSprice()
  p.price = p.BSprice()

  assert c.intrinsic_val() == 0.0
  assert p.intrinsic_val() == 10.0
  assert c.extrinsic_val() == 10.531783818155567
  assert p.extrinsic_val() == 9.709869848260794


def test_IV():
  c = options.CallOption(110, expiry=0.5, vol=.75, und_bid=100, und_ask=100)
  np.testing.assert_allclose(c.IV(10.531783818155567), 0.5, atol=.0001)


def test_greeks():
  c = options.CallOption(110, expiry=.1, und_bid=100, und_ask=100, vol=.5, r=.015)
  p = options.PutOption(110, expiry=0.1, und_bid=100, und_ask=100, vol=.5, r=.015)
  
  assert pytest.approx(c.gamma(), abs=.0001) == .02211
  assert pytest.approx(p.gamma(), abs=.0001) == .02211

  assert pytest.approx(c.delta(), abs=.0001) == .30354
  assert pytest.approx(p.delta(), abs=.0001) == -.69646

  assert pytest.approx(c.vega(), abs=.0001) == 11.05311
  assert pytest.approx(p.vega(), abs=.0001) == 11.05311

  assert pytest.approx(c.theta(), abs=.0001) == -28.04578 # must be per year
  assert pytest.approx(p.theta(), abs=.0001) == -26.39825


def test_r():
  for r, put_val, call_val in zip([.015, .1],
                                  [12.65596, 11.96739],
                                  [2.82083, 3.06191]):
    c = options.CallOption(110, expiry=.1, und_bid=100, und_ask=100, vol=.5, r=r)
    p = options.PutOption(110, expiry=0.1, und_bid=100, und_ask=100, vol=.5, r=r)
    assert pytest.approx(c.BSprice() , call_val, abs=.0001)
    assert pytest.approx(p.BSprice() , put_val, abs=.0001)


def test_deep_itm_call():
  und_price = 1563.555
  expiry = 0.027727499601870865
  opt_price = 1065.869848958933
  c = options.CallOption(750, expiry=expiry, und_price=und_price, opt_price=opt_price)

  print(c.IV())
  return c


def test_find_strike():
  c = options.CallOption(90, 2 / 12, .18, und_price=100)
  print(c.set_strike_from_delta(0.5))
  p0 = c.BSprice()
  c.und_price += .01
  p1 = c.BSprice()
  print((p1 - p0) / .01)
  assert pytest.approx(0.5, .001) == (p1 - p0) / .01
  print(c._strike)
  print(c.und_price)
  print(c._r)