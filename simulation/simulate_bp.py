import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
from logging import getLogger
from statsmodels.tsa.arima_process import ArmaProcess
from matplotlib.dates import DateFormatter

import matplotlib.pyplot as plt

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult, STL
from statsmodels.gam.api import GLMGam, BSplines

from statsmodels.tsa.statespace.sarimax import SARIMAX


logger = getLogger(__name__)


class DecompoeResultBP(DecomposeResult):

    def __init__(self, dti, observed, seasonal, trend, resid, meas_noise=0, weights=None, period=None,
                 true_result=None):

        super().__init__(observed, seasonal, trend, resid, weights=weights)

        self.dti = dti
        self.period = period  # Number of observations per cycle

        self.true_result = true_result
        self._meas_noise = meas_noise

        self.additive_components = {"trend": self.trend, "seasonal": self.seasonal,
                                    "meas_noise": self.meas_noise, "resid": self.resid}

        self.t = np.arange(0, len(dti), 1)
        if period is not None:
            self.t_cycle = 2 * np.pi * self.t / self.period

    @property
    def meas_noise(self):
        """The estimated seasonal component"""
        return self._meas_noise

    def sum(self):
        return sum(self.additive_components.values())

    def df(self):
        return pd.DataFrame({"sum": self.sum()}, index=self.dti)

    def plot(self, date_form=DateFormatter("%m-%d, %H")):
        non_zero_components = {k: v for k, v in self.additive_components.items() if not isinstance(v, int)}
        nrows = len(non_zero_components.keys()) + 1
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(15, 5 * nrows))

        for i, (k, v) in enumerate(non_zero_components.items()):
            l1 = ax[i].plot(self.dti, v, "c-", label="estimated")
            ax[i].title.set_text(k)
            if isinstance(self.true_result, self.__class__):
                l2 = ax[i].plot(self.dti, getattr(self.true_result, k), "m--", label="true")
                ax[i].legend(handles=[l1[0], l2[0]])

        l1 = ax[-1].plot(self.dti, self.sum(), "c-", label="estimated")
        l2 = ax[-1].plot(self.dti, self.observed, "m--", label="observed")
        ax[-1].legend(handles=[l1[0], l2[0]])

        for r in range(nrows):
            ax[r].xaxis.set_major_formatter(date_form)

        plt.tight_layout()
        return fig


def freq_to_period_bp(freq):
    """
    from statsmodels.tsa.tsatools import freq_to_period
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)  # go ahead and standardize
    assert (timedelta(days=1) / freq) % 1 == 0

    return int(timedelta(days=1) / freq)


def linear_trend(t, slope=0, offset=0):
    return offset + (t * slope)


def sinus_seasonal(t, period, seas_apl=10):
    phase_shift = - 0.25 * 2 * np.pi
    return np.sin((2*np.pi * t * (1/period)) + phase_shift) * seas_apl


def evolving_sinus_seasonal(t, period, seas_apl=10):

    n_cycles = t/period
    for c in range(n_cycles):


    phase_shift = - 0.25 * 2 * np.pi
    return np.sin((2*np.pi * t * (1/period)) + phase_shift) * seas_apl


def simulate_bp_simple(start_date=None, ndays=7, freq="0.25H", seed=None,
                       trend_fun=linear_trend, seasonal_fun=sinus_seasonal,
                       arma_scale=1,
                       ar1=np.array([1, -0.5]), ma=np.array([1])):

    if start_date is None:
        start_date = datetime(year=2023, month=1, day=1)

    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, minute=0, second=0,
                          microsecond=0, tzinfo=timezone.utc)

    if seed:
        np.random.seed(seed)

    period = freq_to_period_bp(freq)  # number of observations per cycle, e.g. if freq="H" and cycle 1day --> period=24
    nsample = ndays * period
    dti = pd.date_range(start_date, periods=nsample, freq=freq)
    t = np.arange(0, len(dti), 1)

    trend = trend_fun(t, slope=0.0002)
    seasonality = seasonal_fun(t, period)

    ARMA = ArmaProcess(ar1, ma)
    logger.info(f"{ARMA.isstationary=}")

    arma = ARMA.generate_sample(nsample=nsample, scale=arma_scale)

    meas_noise = np.random.normal(loc=0.0, scale=1, size=nsample)  # what is the meas error
    # meas_noise = 0

    return DecompoeResultBP(dti=dti, trend=trend, seasonal=seasonality, meas_noise=meas_noise, resid=arma,
                            period=period, observed=trend+seasonality+meas_noise+arma)


def decompose(ts_true):
    res = seasonal_decompose(ts_true.df(), period=ts_true.period)
    ts_result = DecompoeResultBP(dti=ts_true.dti, observed=ts_true.sum(), period=ts_true.period, trend=res.trend,
                                 seasonal=res.seasonal, resid=res.resid, true_result=ts_true)
    fig = ts_result.plot()
    fig.savefig("decompose.svg")


def decompose_stl(ts_true):
    stl = STL(ts_true.df(), period=ts_true.period, trend=None, trend_deg=0)
    res = stl.fit()
    ts_result = DecompoeResultBP(dti=ts_true.dti, observed=ts_true.sum(), period=ts_true.period, trend=res.trend,
                                 seasonal=res.seasonal, resid=res.resid, true_result=ts_true)
    fig = ts_result.plot()
    fig.savefig("decompose_stl.svg")


# def decompose_ols(ts_true):
#
#     df = pd.DataFrame()
#     df["t"] = ts_true.t
#     df["sin"] = np.sin(ts_true.t_cycle)
#     df["cos"] = np.cos(ts_true.t_cycle)
#
#     res = sm.OLS(ts_true.sum(), sm.add_constant(df.values))
#     logger.info(res)
#
#     trend = res.partial_values(0, include_constant=True)[0]
#     seasonality = res.params["sin"] * df["sin"].values + res.params["cos"] * df["cos"].values
#     residual = ts_true.sum() - trend - seasonality
#
#     ts_result = DecompoeResultBP(dti=ts_true.dti, observed=ts_true.sum(), period=ts_true.period, trend=trend,
#                                  seasonal=seasonality, resid=residual, true_result=ts_true)
#     fig = ts_result.plot()
#     fig.savefig("decompose_gam.svg")


def decompose_gam(ts_true, linear_trend=False, figname="decompose_gam.svg"):

    df = ts_true.df()

    df["sin"] = np.sin(ts_true.t_cycle)
    df["cos"] = np.cos(ts_true.t_cycle)
    df["t"] = ts_true.t

    if linear_trend:
        # TO only fit constant trend
        bs = BSplines(ts_true.t, df=1, degree=0, constraints=None, include_intercept=False)
        alpha = 0.9

        gam_bs = GLMGam.from_formula('sum ~ t + sin + cos', data=df, smoother=bs, alpha=alpha)
    else:
        bs = BSplines(ts_true.t, df=10, degree=3, constraints=None, include_intercept=False)
        alpha = 0.9
        gam_bs = GLMGam.from_formula('sum ~ sin + cos', data=df, smoother=bs, alpha=alpha)

    res = gam_bs.fit()  # default is pirls with use_t=False
    logger.info(res)

    if linear_trend:
        trend = res.params["t"] * df["t"].values + res.params["Intercept"]
    else:
        trend = res.partial_values(0, include_constant=True)[0]

    seasonality = res.params["sin"] * df["sin"].values + res.params["cos"] * df["cos"].values
    residual = ts_true.sum() - trend - seasonality

    ts_result = DecompoeResultBP(dti=ts_true.dti, observed=ts_true.sum(), period=ts_true.period, trend=trend,
                                 seasonal=seasonality, resid=residual, true_result=ts_true)
    fig = ts_result.plot()
    fig.savefig(figname)


if __name__ == "__main__":

    ts_true = simulate_bp_simple()
    decompose(ts_true)
    decompose_stl(ts_true)
    decompose_gam(ts_true)
    decompose_gam(ts_true, linear_trend=True, figname="decompose_gam_lin_trend.svg")





