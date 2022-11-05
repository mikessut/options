from matplotlib.widgets import MultiCursor
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from options import egarch


def test_egarch():
    df = pd.read_csv(pathlib.Path(__file__).parent / "SPY_20171104-20221104.csv")
    df.index = pd.DatetimeIndex(df.Date) 
    lr = np.log(df.Close / df.Close.shift())[1:]

    X = egarch.fit(lr)

    print(X)

    f, ax = plt.subplots(3, 1, sharex=True)
    
    ax[0].plot(df.Close)
    ax[1].plot(lr)
    ax[2].plot(pd.Series(np.sqrt(egarch.calc_egarch(lr, *X.x) * 252), lr.index), label='egarch')
    multi = MultiCursor(f.canvas, ax, horizOn=True)

    plt.show()


def test_egarch_monte_carlo():
    strikes = [95, 100, 105]
    g = egarch.EGARCHMonteCarlo(101.1, 0, strikes, 20, .2**2 / 252,
            -0.54267002,  0.93927269, -0.16669602,  0.31388232)

    g.run()

    for K in strikes:
        print(g.call(K, 20), g.put(K, 20))