Options package that performs basic Black Scholes calculations.

## Usage:

```
from options import CallOption

c = CallOption(strike=105, expiry=.2, vol=.5, und_price=100)
print(c.BSprice())
print(c.extrinsic_val())  # Since bid/ask not specified, this returns nan.
print(c.extrinsic_val('BSprice'))

# Determine IV
c = CallOption(strike=105, expiry=.2, und_price=100, bid=8, ask=9)
print(c.IV(c.mid()))
```

## Tests

Run `pytest` to run test suite.