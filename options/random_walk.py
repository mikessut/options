import numpy as np
import matplotlib.pyplot as plt


def corr_rand(rho, size=None):
    cov_mat = np.array([[1, rho],
                        [rho, 1]])
    return np.random.multivariate_normal([0, 0], cov_mat, size)


class StochasticVolBase:

    def __init__(self, var0, price0):
        self._var = [var0]
        self._price = [price0]

    def nsteps(self, num):
        for _ in range(num):
            self.step()

    def realized_vol(self):
        ret = np.array(self._price[1:]) / np.array(self._price[:-1])
        lr = np.log(ret)
        return np.std(lr)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(np.sqrt(np.array(self._var)))
        ax[0].set_ylabel('Vol')
        # ax[0].axhline(np.array(self._vol0) * vol_sf, color='k', linestyle='--')
        ax[0].grid(True)

        ax[1].plot(self._price)
        ax[1].set_ylabel('Price')
        ax[1].grid(True)
        return ax


class StocasticVol(StochasticVolBase):

    def __init__(self, vol0, price0, kappa, vol_vol):
        """
        :param vol0: Long term volatility
        :param price0: Initial price
        :param kappa: How strongly vol returns to long term average
        :param vol_vol: Volatility of volatility
        """
        super().__init__(vol0, price0)

        self._vol0 = vol0
        self._kappa = kappa
        self._vol_vol = vol_vol

    def step(self):
        dvol = self._kappa * (self._vol0 - self._vol[-1]) + self._vol_vol * np.random.randn()
        vol = self._vol[-1] + dvol
        dprice = self._price[-1] * vol * np.random.randn()
        price = self._price[-1] + dprice

        self._vol.append(vol)
        self._price.append(price)



class SVJ(StochasticVolBase):
    """
    Gatheral pg. 65
    dS = mu*S*dt + sqrt(v)*S*dZ1 + (exp(alpha+delta*eps) - 1)*S*dq
    dv = -lam*(v-v_avg)*dt + eta*sqrt(v)*dZ2

    lamJ [=] jumps / year
    """

    def __init__(self, und_price, lam, eta, rho, v_avg, v, lamJ, alpha, delta, mu, dt):
        super().__init__(v, und_price)
        
        self._lam = lam
        self._eta = eta
        self._rho = rho
        self._v_avg = v_avg
        self._lamJ = lamJ
        self._alpha = alpha
        self._delta = delta
        self._dt = dt
        self._sqrt_dt = np.sqrt(dt)

        self._mu = mu

        self._min_var = .01**2

        self._poisson_err = 0
        self._num_jumps = 0

    def step(self):
        Z1, Z2 = corr_rand(self._rho)

        dvar = -self._lam * (self._var[-1] - self._v_avg) * self._dt \
               +self._eta * np.sqrt(self._var[-1]) * self._sqrt_dt * Z2

        var = self._var[-1] + dvar
        var = max(self._min_var, var)
        jump = np.random.poisson(self._lamJ * self._dt)
        if jump > 1:
            # This means we are trying to model multiple jumps per time step.
            self._poisson_err += 1

        dprice = self._mu * self._price[-1] * self._dt + np.sqrt(var) * self._sqrt_dt * self._price[-1] * Z1
        if jump > 0:
            self._num_jumps += 1
            dprice += (np.exp(self._alpha + self._delta * np.random.randn()) - 1) * self._price[-1]

        price = self._price[-1] + dprice
        self._var.append(var)
        self._price.append(price)