import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult, STL
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # start 1966, frequency = 12
    hstart = pd.read_csv("hstart.dat", header=None)
    start_date = datetime(year=1966, month=1, day=31, tzinfo=timezone.utc)
    frequency = 12  # number of observations per cycle, i.e. 12 month per year

    dti = pd.date_range(start_date, periods=len(hstart), freq="M")

    hstart.set_index(dti, inplace=True)
    #
    # time = [datetime(year=start_date.year + int(m/12), month=start_date.month + m % 12, day=1) for m in
    #         range(0, len(hstart))]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(hstart)
    fig.show()

    #  Seasonal decomposition using moving averages. Average is over whole period, i.e. average
    # over - 6 month to + 6 month.
    ts_dicomposition = seasonal_decompose(hstart, period=12)

    fig = ts_dicomposition.plot()
    fig.show()
    stl = STL(hstart, seasonal=12)
    res = stl.fit()

    #
    # fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    # fig.set_figheight(10)
    # fig.set_figwidth(20)
    # # First plot to the Original time series
    # axes[0].plot(hstart, label='Original')
    # axes[0].legend(loc='upper left')
    # # second plot to be for trend
    # axes[1].plot(ts_dicomposition.trend, label='Trend')
    # axes[1].legend(loc='upper left')
    # # third plot to be Seasonality component
    # axes[2].plot(ts_dicomposition.seasonal, label='Seasonality')
    # axes[2].legend(loc='upper left')
    # # last last plot to be Residual component
    # axes[3].plot(ts_dicomposition.resid, label='Residuals')
    # axes[3].legend(loc='upper left')
    #
    # fig.show()



