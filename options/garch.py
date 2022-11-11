import arch
import numpy as np
from scipy.optimize import minimize
from options import portfolio
import options
import logging
import datetime

log = logging.getLogger(__name__)


def calc_garch(w, alpha, beta, lr, var0=None):
    var = np.zeros((len(lr),))
    if var0 is None:
        # https://stats.stackexchange.com/questions/367722/fitting-a-garch1-1-model
        var[0] = w / (1 - alpha - beta)
    else:
        var[0] = var0
    for n in range(1, len(lr)):
        var[n] = w + alpha * lr[n-1]**2 + beta * var[n-1]
    var[var < 1e-12] = 1e-12
    return var


def fit_garch(historical_price, var0):
    """
    https://web-static.stern.nyu.edu/rengle/GARCH101.PDF
    https://www.youtube.com/watch?v=Xef4LL2KUy0

    r(t) = m + sqrt(h) * err
    r: return
    m: long term average
    h: variance of return
    err: standard normal distribution

    GARCH(1,1) Model
    sigma_n^2 = w + alpha * lr_(n-1)^2 + beta * sigma_(n-1)^2
    sigma^2 = var(iance)

    Use maximum likelihood approach, which is to adjust parameters to maximize likelihood.

    Likelihood defined as: -log(sigma^2) - ret^2 / sigma^2
    """

    lr = np.log(historical_price[1:] / historical_price[:-1])

    lr -= lr.mean()

    def neg_likelihood(X, var0, lr):
        w, alpha, beta = X
        w = w / 100
        #alpha = alpha / 10
        var = calc_garch(w, alpha, beta, var0, lr)
        return -np.sum(-np.log(var) - (lr)**2 / var)

    returns_sf = 100

    X = minimize(neg_likelihood, (0.1*100, .05, .8), 
                args=(var0, lr * returns_sf),
                #bounds=((0.0001, None), (0.0001, None), (0.0001, None)),
                method='SLSQP',
                options={'disp': True})
    return X.x[0] / returns_sf**2/100, X.x[1], X.x[2]   # w, alpha, beta


def fit_garch2(historical_price):
    """
    """
    lr = np.log(historical_price[1:] / historical_price[:-1])
    # sr = 1 - historical_price[1:] / historical_price[:-1]
    # Apparently this package want return in % so multiply by 100
    garch = arch.arch_model(lr, vol='garch', p=1, o=0, q=1, rescale=True)
    res = garch.fit()
    return res.params['omega'] / res.scale**2, res.params['alpha[1]'], res.params['beta[1]']


class GARCHMonteCarlo:

    def __init__(self, p0, strikes, days_to_expiration, var0, w, alpha, beta, 
                 r=0.015, mu=0, num_sims=5000,
                 days_in_year=365.25):
        """
        :param p0: Initial underlying price
        :param strikes: List of strike prices
        :param days_to_expiration: List of days to expiration.  These are really steps in the 
                             simulation.  days_in_year is used to convert to years.
        :param var0: Initial variance in the GARCH simulation
        :param w:
        :param a:
        :param b:
        :param r: Annualized discount rate
        :param mu: Daily expected return
        :param num_sims: Number of simulations to perform
        :param days_in_year: Used to convert time steps to years for discounting.
        """
        self._p0 = p0
        self._strikes = np.array(strikes, ndmin=1)
        self._days_to_expiration = np.array(days_to_expiration, ndmin=1, dtype=int)
        self._price_paths = np.zeros((max(self._days_to_expiration), num_sims))
        self._lrs = np.zeros((max(self._days_to_expiration), num_sims))
        self._put_prices = np.zeros((len(self._strikes), len(self._days_to_expiration), num_sims))
        self._call_prices = np.zeros((len(self._strikes), len(self._days_to_expiration), num_sims))

        self._var0 = var0
        self._w = w
        self._alpha = alpha
        self._beta = beta
        self._long_term_annual_vol = np.sqrt((self._w / (1 - self._alpha - self._beta)) * 365.25)
        self._mu = mu
        self._r = r
        self._num_sims = num_sims
        self._days_in_year = days_in_year

        self._min_vol_ratio = .25  # Guard to not allow vol below sqrt(var0)

        self._has_run = False

    def run(self):
        log.info(f"GARCHMonteCarlo Run Started.")
        self._create_lrs()
        self._create_price_paths()
        self._update_opt_prices()
        self._has_run = True
        log.info(f"GARCHMonteCarlo Run complete.")

    def set_und_price(self, price):
        self._p0 = price        
        self._update_opt_prices()

    @property
    def und_price(self):
        return self._p0

    def _create_lrs(self):
        for n in range(self._num_sims):
            var = self._var0
            p_prev = 1.0
            for nd in range(max(self._days_to_expiration)):
                p = p_prev * (1 + self._mu + np.sqrt(var) * np.random.randn())
                #p = p_prev * np.exp(self._mu + np.sqrt(var) * np.random.randn()) # This doesn't seem to match BS as well
                sr = p / p_prev - 1  # or should this be lr?
                lr = np.log(p / p_prev)
                var = self._w + self._alpha * sr**2 + self._beta * var
                p_prev = p
                self._lrs[nd, n] = lr
                # self._price_paths[nd, n] = p

    def _create_price_paths(self):
        self._price_paths = np.exp(self._lrs.cumsum(axis=0))
        # for n in range(self._num_sims):
        #     for nd in range(max(self._days_to_expiration)):
        #         self._price_paths[nd, n] = np.exp(self._lrs[:nd+1, n].sum())

    def _update_opt_prices(self):
        # self._strikes_pct = self._strikes / self._p0
        for n_strike in range(len(self._strikes)):
            for i, n_days in enumerate(self._days_to_expiration):
                # index into self._price_paths is n_day - 1 (e.g. 1 days is 0th index)

                t = n_days / self._days_in_year
                # Forward price of the underlying asset
                F = np.exp(self._r * t) * self._p0
                # Calls
                idx_itm = self._price_paths[n_days-1, :] * F > self._strikes[n_strike]
                idx_otm = self._price_paths[n_days-1, :] * F <= self._strikes[n_strike]
                self._call_prices[n_strike, i, idx_itm] = (self._price_paths[n_days-1, idx_itm] * F - self._strikes[n_strike])
                self._call_prices[n_strike, i, idx_otm] = 0

                # Puts
                idx_itm = self._price_paths[n_days-1, :] * F < self._strikes[n_strike]
                idx_otm = self._price_paths[n_days-1, :] * F >= self._strikes[n_strike]
                self._put_prices[n_strike, i, idx_itm] = (self._strikes[n_strike] - self._price_paths[n_days-1, idx_itm] * F)
                self._put_prices[n_strike, i, idx_otm] = 0

    def _add_new_days_to_expiration(self, days):
        self._days_to_expiration = np.append(self._days_to_expiration, days)
        self._put_prices = np.zeros((len(self._strikes), len(self._days_to_expiration), self._num_sims))
        self._call_prices = np.zeros((len(self._strikes), len(self._days_to_expiration), self._num_sims))
        self._update_opt_prices()

    def put(self, strike, days: int):
        if not self._has_run:
            raise ValueError("Trying to get price without running simulation.")
        idx_strike = np.where(self._strikes == strike)[0][0]
        if days not in self._days_to_expiration and days < max(self._days_to_expiration):
            self._add_new_days_to_expiration(days)
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        t = days / self._days_in_year
        model_price = self._put_prices[idx_strike, idx_days, :].mean() * np.exp(-self._r * t)
        # vol_annual = np.sqrt(self._var0*365.25) * self._min_vol_ratio
        expiry = days / self._days_in_year
        put = options.PutOption(strike, expiry, self._long_term_annual_vol * self._min_vol_ratio, und_price=self._p0, r=self._r)
        bs_price = put.BSprice()
        if bs_price > model_price:
            log.debug(f"garch.put() BS price is larger than model price. put priced at: {model_price:.3f} vol: {self._long_term_annual_vol * self._min_vol_ratio:.3f} BSprice: {bs_price:.4f} t: {days/365.25:.5f} expiry: {expiry}")
            return bs_price
        return model_price

    def call(self, strike, days: int):
        if not self._has_run:
            raise ValueError("Trying to get price without running simulation.")
        idx_strike = np.where(self._strikes == strike)[0][0]
        if days not in self._days_to_expiration and days < max(self._days_to_expiration):
            self._add_new_days_to_expiration(days)
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        t = days / self._days_in_year
        # vol_annual = np.sqrt(self._var0*365.25) * self._min_vol_ratio
        model_price = self._call_prices[idx_strike, idx_days, :].mean() * np.exp(-self._r * t)
        expiry = days / self._days_in_year
        call = options.CallOption(strike, expiry, self._long_term_annual_vol * self._min_vol_ratio, und_price=self._p0, r=self._r)
        bs_price = call.BSprice()
        if bs_price > model_price:
            log.debug(f"garch.call() BS price is larger than model price. call priced at: {model_price:.3f} vol: {self._long_term_annual_vol * self._min_vol_ratio:.3f} BSprice: {bs_price:.4f} t: {days/365.25:.5f} expiry: {expiry}")
            return bs_price
        return model_price

    def portfolio_expected_value(self, prt: 'portfolio.Portfolio'):
        prt_returns = np.zeros((self._num_sims,))
        for pos in prt:
            if isinstance(pos, portfolio.OptionPosition):
                if isinstance(pos.option, options.PutOption):
                    idx_strike = np.where(self._strikes == pos.option.strike)[0][0]
                    idx_days = np.where(self._days_to_expiration == int(np.round(pos.option.t_expiry()*365.25)))[0][0]
                    prt_returns += self._put_prices[idx_strike, idx_days, :] * pos.option.multiplier * pos.qty
                elif isinstance(pos.option, options.CallOption):
                    idx_strike = np.where(self._strikes == pos.option.strike)[0][0]
                    idx_days = np.where(self._days_to_expiration == int(np.round(pos.option.t_expiry()*365.25)))[0][0]
                    prt_returns += self._call_prices[idx_strike, idx_days, :] * pos.option.multiplier * pos.qty
            else:
                print("Not handling non-option positions.....")
        return prt_returns.mean(), prt_returns.std()

    def call_pop(self, basis, strike, days):
        """
        This assumes a "long" position. To determine the corresponding short
        position subtract from 1.  E.g.:
        Short_position_pop = 1 - call_pop(premium_rcvd, strike, days)

                │               │   /
                │               │  /
                │               │ /
                │               │/
             0  ├───────────────┼──────────
                │              /│
                │             / │
                ├────────────/  │ POP Area
                │               │

        """
        idx_strike = np.where(self._strikes == strike)[0][0]
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        idx = self._call_prices[idx_strike, idx_days, :] > basis
        return sum(idx) / self._num_sims

    def put_pop(self, basis, strike, days):
        """
        This assumes a "long" position. To determine the corresponding short
        position subtract from 1.  E.g.:
        Short_position_pop = 1 - put_pop(premium_rcvd, strike, days)
        """
        idx_strike = np.where(self._strikes == strike)[0][0]
        idx_days = np.where(self._days_to_expiration == days)[0][0]
        idx = self._put_prices[idx_strike, idx_days, :] > basis
        return sum(idx) / self._num_sims

    def pop(self, pos):
        if isinstance(pos, portfolio.OptionPosition):
            return self._opt_pop(pos)
        elif isinstance(pos, portfolio.Portfolio):
            return self._portfolio_pop(pos)

    def _portfolio_pop(self, prt: 'portfolio.Portfolio'):
        prt_returns = np.zeros((self._num_sims,))
        for pos in prt:
            if isinstance(pos, portfolio.OptionPosition):
                if isinstance(pos.option, options.PutOption):
                    idx_strike = np.where(self._strikes == pos.option.strike)[0][0]
                    idx_days = np.where(self._days_to_expiration == int(np.round(pos.option.t_expiry()*365.25)))[0][0]
                    prt_returns += self._put_prices[idx_strike, idx_days, :] * pos.option.multiplier * pos.qty
                elif isinstance(pos.option, options.CallOption):
                    idx_strike = np.where(self._strikes == pos.option.strike)[0][0]
                    idx_days = np.where(self._days_to_expiration == int(np.round(pos.option.t_expiry()*365.25)))[0][0]
                    prt_returns += self._call_prices[idx_strike, idx_days, :] * pos.option.multiplier * pos.qty
            else:
                print("Not handling non-option positions.....")
        basis = 0  # amount it _cost_ to enter position
        for pos in prt:
            if isinstance(pos, portfolio.OptionPosition):
                multiplier = pos.option.multiplier
            else: 
                multiplier = 1.0
            basis += pos.basis * pos.qty * multiplier
        print(f"basis: {basis} max return: {max(prt_returns - basis)} min return {min(prt_returns - basis)}")
        idx = prt_returns > basis
        return sum(idx) / self._num_sims

    def _opt_pop(self, opt_pos: 'portfolio.OptionPosition'):
        if isinstance(opt_pos.option, options.PutOption):
            if opt_pos.qty > 0:
                return self.put_pop(opt_pos.basis, opt_pos.option.strike, int(np.round(opt_pos.option.t_expiry()*365.25)))
            else:
                # short position
                # assert opt_pos.basis < 0, "Something doesn't seem right. Qty is negative but basis is positive???"
                return 1 - self.put_pop(opt_pos.basis, opt_pos.option.strike, int(np.round(opt_pos.option.t_expiry()*365.25)))
        elif isinstance(opt_pos.option, options.CallOption):
            if opt_pos.qty > 0:
                return self.call_pop(opt_pos.basis, opt_pos.option.strike, int(np.round(opt_pos.option.t_expiry()*365.25)))
            else:
                # short position
                # assert opt_pos.basis < 0, "Something doesn't seem right. Qty is negative but basis is positive???"
                return 1 - self.call_pop(opt_pos.basis, opt_pos.option.strike, int(np.round(opt_pos.option.t_expiry()*365.25)))

    @staticmethod
    def garch_monte_carlo(p0, strikes, days_to_expiration,
                          var0, w, alpha, beta, mu=0, num_sims=1000,
                          return_avgs=True):

        strikes = np.array(strikes, ndmin=1)
        days_to_expiration = np.array(days_to_expiration, ndmin=1, dtype=int)
        put_prices = np.zeros((len(strikes), len(days_to_expiration), num_sims))
        call_prices = np.zeros((len(strikes), len(days_to_expiration), num_sims))
        for n in range(num_sims):
            p_prev = p0
            var = var0
            for nd in range(max(days_to_expiration)):
                p = p_prev * (1 + mu + np.sqrt(var) * np.random.randn())
                sr = 1 - p / p_prev
                var = w + alpha * sr**2 + beta * var
                p_prev = p
                if (nd+1) in days_to_expiration:
                    idx = np.where(nd+1 == days_to_expiration)[0]
                    for m, strike in enumerate(strikes):
                        put_price = 0 if p > strike else strike - p
                        call_price = 0 if p < strike else p - strike
                        put_prices[m, idx, n] = put_price
                        call_prices[m, idx, n] = call_price
        if return_avgs:
            return np.mean(call_prices, axis=2), np.mean(put_prices, axis=2)
        else:
            return call_prices, put_prices


class GARCHMonteCarloEarnings(GARCHMonteCarlo):

    def __init__(self,
            p0, strikes, days_to_expiration, var0, w, alpha, beta, 
            earnings_move_std: float, earnings_days_from_now: list[int],
            r=0.015, mu=0, num_sims=5000, days_in_year=365.25
        ):
        super().__init__(p0, strikes, days_to_expiration, var0, w, alpha, beta, 
            r=0.015, mu=mu, num_sims=num_sims, days_in_year=days_in_year)
        self._earnings_move_std = earnings_move_std
        self._earnings_days_from_now = earnings_days_from_now

    def _create_lrs(self):
        super()._create_lrs()
        # Now go in and add earnings moves
        for n in range(self._num_sims):
            for day in self._earnings_days_from_now:
                self._lrs[day-1, n] += np.random.randn() * self._earnings_move_std

