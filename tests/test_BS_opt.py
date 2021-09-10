"""
Good calculator to check:
https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
"""
import options
import datetime
import numpy as np
import pytest


def test_BS_value():
  c = options.CallOption(110, datetime.datetime(2021, 10, 15), .5, 100)
  p = options.PutOption(110, datetime.datetime(2021, 10, 15), .5, 100)
  assert c.BSprice(t=.5) == 10.531783818155567
  assert p.BSprice(t=.5) == 19.709869848260794

  assert c.intrinsic_val() == 0.0
  assert p.intrinsic_val() == 10.0
  assert c.extrinsic_val('BSprice', t=.5) == 10.531783818155567
  assert p.extrinsic_val('BSprice', t=0.5) == 9.709869848260794


def test_IV():
  c = options.CallOption(110, datetime.datetime(2021, 10, 15), .75, 100)
  np.testing.assert_allclose(c.IV(10.531783818155567, t=.5), 0.5, atol=.0001)


def test_greeks():
  c = options.CallOption(110, expiry=.1, und_price=100, vol=.5, r=.015)
  p = options.PutOption(110, expiry=0.1, und_price=100, vol=.5, r=.015)
  
  assert pytest.approx(c.gamma(), abs=.0001) == .02211
  assert pytest.approx(p.gamma(), abs=.0001) == .02211

  assert pytest.approx(c.delta(), abs=.0001) == .30354
  assert pytest.approx(p.delta(), abs=.0001) == -.69646

  assert pytest.approx(c.vega(), abs=.0001) == 11.05311
  assert pytest.approx(p.vega(), abs=.0001) == 11.05311

  assert pytest.approx(c.theta(), abs=.0001) == -28.04578 # must be per year
  assert pytest.approx(p.theta(), abs=.0001) == -26.39825