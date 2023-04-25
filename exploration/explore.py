import numpy as np
import itertools
from simulation.simulate_bp import BPTimseSeriesSimulator, random_cosinus_seasonal, cosinus_seasonal, linear_trend
from functools import partial
from exploration.blr import blr_corr, plot_blr_output, blr_simple
import matplotlib.pyplot as plt
from exploration.gp import GPModel, ARKernel
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_acovf
from log_setup import setup_logging
from logging import getLogger

logger = getLogger(__name__)


def get_red_idx(n, data_fraction=0.3, weights=None):
    if data_fraction == 1:
        return range(n)
    k = int(n * data_fraction)
    if weights is not None:
        weights = weights/sum(weights)
    return sorted(np.random.choice(range(n), size=k, replace=False, p=weights))


def get_sesonal_weights(n, period, phase=np.pi):
    weights = cosinus_seasonal(range(n), period, phase=phase)
    return weights - min(weights)


def get_linear_weights(n, slope=1):
    return linear_trend(range(n), slope=slope)


def get_binary_weights(n, split=None, prob=0.1):
    default_prob = 0.9
    weights = np.array(n*[default_prob])
    if split is None:
        split = int(n/2)
    weights[split:] = default_prob * prob
    return weights


def get_default_config():
    default_config = {
        "rng": np.random.default_rng(seed=10),
        "ndays": 3,
        "samples_per_hour": 5
    }
    return default_config


def find_optimal_length_scale(cor_lag_1, kernel, len_scale_choices=np.arange(1,10)):
    cal_x = np.array([0, 1]).reshape(-1, 1)
    result = {}
    for len_scale in len_scale_choices:
        result[len_scale] = np.abs(kernel(length_scale=len_scale)(cal_x)[0, 1] - cor_lag_1)
    return min(result, key=result.get)


def simulate_plot_ar(ar, wn_scale, data_fraction=0.1):
    ar_order = len(ar) - 1
    s1_config = {"ar": ar, "arma_scale": wn_scale,  "ma": np.array([1])}
    s1 = BPTimseSeriesSimulator(**s1_config, **get_default_config())
    ts1 = s1.generate_sample()
    fig = ts1.plot()
    fig.savefig(f"true_ts_ar{ar_order}_{data_fraction}.svg")
    plt.show()

    cov_true = arma_acovf(ar=s1_config["ar"], ma=s1_config["ma"], sigma2=s1_config["arma_scale"], nobs=s1.nsample)

    # This is only true for AR1
    var_true = wn_scale/(1-ar[1]**2)
    logger.info(f"{cov_true[0]=}, {var_true=}")

    # arima = ARIMA(ts1.sum(), order=(1, 0, 0), trend_offset=0)
    # s1fit = arima.fit(gls=False)
    # ar_est = np.array([1, -s1fit.params[1]])
    # cov_est = arma_acovf(ar=ar_est, ma=s1_config["ma"], sigma2=s1fit.params[2], nobs=s1.nsample)

    cov_true_matrix = np.zeros(shape=(s1.nsample, s1.nsample))
    for (i, j) in itertools.product(range(s1.nsample), range(s1.nsample)):
        cov_true_matrix[i, j] = cov_true[i-j]
        cov_true_matrix[j, i] = cov_true[i-j]

    kernel_ = partial(ARKernel, order=ar_order)
    length_scale = find_optimal_length_scale(cov_true[1]/cov_true[0], kernel_)
    logger.info(f"best {length_scale=}")

    mean_prior = np.zeros(ts1.t.shape[0])
    kernel = cov_true[0] * kernel_(length_scale=length_scale)
    cov_prior = kernel(s1.t.reshape(-1, 1))
    std_prior = np.sqrt(np.diag(cov_prior))
    gpm = GPModel(kernel=kernel, normalize_y=False, meas_noise=0)

    idx_train = get_red_idx(len(ts1.t), data_fraction=data_fraction)
    x_train = ts1.t[idx_train]
    y_train = ts1.sum()[x_train]
    gpm.fit_model(x_train.reshape(-1, 1), y_train)
    mean_post, cov_post = gpm.predict(ts1.t.reshape(-1, 1), return_cov=True)
    std_post = np.sqrt(np.diag(cov_post))

    cov_opt = gpm.gp.kernel_(s1.t.reshape(-1, 1))

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 30))
    ax[0, 0].plot(s1.t[:50], cov_true[:50])
    ax[0, 0].set_title(f"Theoretical Cov for {ar=}")
    ax[1, 0].plot(s1.t[:50], cov_prior[0, :50])
    ax[1, 0].set_title(f"Initial Prior Cov {kernel=}")
    ax[2, 0].plot(s1.t[:50], cov_post[:50])
    ax[2, 0].set_title(f"Posterior Cov")
    ax[3, 0].plot(s1.t[:50], cov_opt[0, :50])
    ax[3, 0].set_title(f"Optimal Prior Cov {gpm.gp.kernel_=}")

    cmap = 'viridis'
    if min(cov_true) < 0:
        cmap = "PiYG"
    im1 = ax[0, 1].imshow(cov_true_matrix[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    im2 = ax[1, 1].imshow(cov_prior[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    im3 = ax[2, 1].imshow(cov_post[:50, :50], cmap=cmap)
    im4 = ax[3, 1].imshow(cov_opt[:50, :50], cmap=cmap)

    fig.colorbar(im1)
    fig.colorbar(im2)
    fig.colorbar(im3)
    fig.colorbar(im4)
    fig.savefig(f"covariance_ar{ar_order}_{data_fraction}.svg")
    plt.show()

    n = 100
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].plot(ts1.t, np.transpose(np.random.multivariate_normal(mean=mean_prior, cov=cov_prior, size=n)),  "b-", alpha=0.1)
    ax[0].plot(ts1.t, ts1.sum(), 'r-', label="true", alpha=0.8)
    ax[0].set_title(f"{n=} samples from prior")
    ax[1].plot(x_train, y_train, 'yo', label="observations", markersize=12)
    ax[1].plot(ts1.t, mean_post,  "b-", label="posterior")
    ax[1].fill_between(ts1.t, mean_post-std_post, mean_post + std_post,
                     color='b', alpha=0.2, label='CI')
    ax[1].plot(ts1.t, ts1.sum(), 'r-', label="true", alpha=0.5)
    plt.legend()
    fig.savefig(f"regression_line_ar{ar_order}_{data_fraction}.svg")
    plt.show()

    return s1_config


if __name__ == "__main__":
    setup_logging()

    # Simulate AR(1) with phi = 0.8
    phi = 0.8
    wn_scale = 1

    ar = np.array([1, -phi])
    s1_config = simulate_plot_ar(ar, wn_scale, data_fraction=.1)

    ar = np.array([1, -phi/2, -phi/3])
    s2_config = simulate_plot_ar(ar, wn_scale, data_fraction=.1)









    # Simulate AR(1) with seasonality
    s2_config = {"seasonal_fun": partial(cosinus_seasonal, seas_ampl=10), **s1_config}
    s2 = BPTimseSeriesSimulator(**s2_config, **get_default_config())
    ts2 = s2.generate_sample()
    ts2.plot()
    plt.show()



    #
    #
    #
    # data_fraction = 0.1
    #
    # simulation_config = dict(
    #     seas_ampl=2,
    #     ndays=4,
    #     samples_per_hour=10)
    #
    # # ts_true, true_regression_line = simulate_random_seasonal(**simulation_config)
    # ts_true, true_regression_line = simulate_cos_seasonal(**simulation_config)
    #
    # red_idx = get_red_idx(len(ts_true.t), data_fraction=data_fraction,
    #                       weights=get_binary_weights(len(ts_true.t)))
    # t = ts_true.t[red_idx]
    # X = get_design_matrix(ts_true.t[red_idx], ts_true.period)
    # y = ts_true.sum()[red_idx]
    #
    # gp_fitted, gp = fit_gp(t.reshape(-1, 1), y, period=ts_true.period)
    # cov = np.dot(gp_fitted.L_, np.transpose(gp_fitted))
    # cov = gp_fitted.kernel_
    #
    # gp_mean, gp_std = gp.predict(ts_true.t.reshape(-1, 1), return_std=True)
    # fig, ax = plt.subplots()
    # ax.plot(ts_true.t, gp_mean,  "b-", label="predicted")
    # plt.fill_between(ts_true.t, gp_mean-gp_std, gp_mean + gp_std,
    #                  color='b', alpha=0.2, label='CI')
    # ax.plot(t, y, 'y+', alpha=0.8, label="observations")
    #
    # ax.plot(ts_true.t, ts_true.trend + ts_true.seasonal + ts_true.resid, 'r-', label="true")
    # plt.legend()
    # plt.show()
    #
    #
    #
    # # idata = blr_simple(X, y)
    # # fig = plot_blr_output(idata, ts_true.t[red_idx], ts_true.t, true_regression_line)
    # # fig.savefig("blr_simple.svg")
    # # idata = blr_corr(X, y)
    #

