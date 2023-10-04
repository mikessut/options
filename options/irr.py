"""
dates = [
    '4/4/16',
    '7/1/16',
    '4/5/16',
    '4/19/16',
]

vals = np.array([
    150500,
-152306.07,
-25000,
25000,
])


dates = [datetime.datetime.strptime(x, '%m/%d/%y') for x in dates]

print(irr(dates, vals))
"""
import numpy as np
import datetime
from scipy.optimize import root_scalar
from typing import List


def irr(dates: List[datetime.datetime], vals: np.array):
    maxdate = min(dates)
    t = np.array([(maxdate - x).total_seconds() / 3600 / 24 / 365.25 for x in dates])

    def err(r):
        FV = vals * (1 + r) ** t
        return sum(FV)

    X = root_scalar(err, x0=.2, x1=-.2)
    return X.root
