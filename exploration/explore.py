import numpy as np
import itertools
from simulation.simulate_bp import BPTimseSeriesSimulator, random_cosinus_seasonal, cosinus_seasonal, linear_trend
from functools import partial
from exploration.blr import blr_corr, plot_blr_output, blr_simple
import matplotlib.pyplot as plt
from exploration.gp import GPModel, get_AR_kernel
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_acovf


def get_red_idx(n, data_fraction=0.3, weights=None):
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


def simulate_plot_ar(ar, wn_scale, data_fraction=0.1):
    ar_order = len(ar) - 1
    s1_config = {"ar": ar, "arma_scale": wn_scale,  "ma": np.array([1])}
    s1 = BPTimseSeriesSimulator(**s1_config, **get_default_config())
    ts1 = s1.generate_sample()
    ts1.plot()
    plt.show()

    cov_true = arma_acovf(ar=s1_config["ar"], ma=s1_config["ma"], sigma2=s1_config["arma_scale"], nobs=s1.nsample)

    # This is only true for AR1
    var_true = wn_scale/(1-phi**2)
    print(f"{cov_true[0]=}, {var_true=}")

    # arima = ARIMA(ts1.sum(), order=(1, 0, 0), trend_offset=0)
    # s1fit = arima.fit(gls=False)
    # ar_est = np.array([1, -s1fit.params[1]])
    # cov_est = arma_acovf(ar=ar_est, ma=s1_config["ma"], sigma2=s1fit.params[2], nobs=s1.nsample)

    cov_true_matrix = np.zeros(shape=(s1.nsample, s1.nsample))
    for (i, j) in itertools.product(range(s1.nsample), range(s1.nsample)):
        cov_true_matrix[i, j] = cov_true[i-j]
        cov_true_matrix[j, i] = cov_true[i-j]

    mean_prior = np.zeros(ts1.t.shape[0])
    cov_prior = (cov_true[0] * get_AR_kernel(order=ar_order, length_scale=5))(s1.t.reshape(-1, 1))
    std_prior = np.sqrt(np.diag(cov_prior))
    gpm = GPModel(kernel=cov_true[0] * get_AR_kernel(order=ar_order, length_scale=5), normalize_y=False, meas_noise=0)

    idx_train = get_red_idx(len(ts1.t), data_fraction=data_fraction)
    x_train = ts1.t[idx_train]
    y_train = ts1.sum()[x_train]
    gpm.fit_model(x_train.reshape(-1, 1), y_train)
    mean_post, cov_post = gpm.predict(ts1.t.reshape(-1, 1), return_cov=True)
    std_post = np.sqrt(np.diag(cov_post))
    print(f"Posterior kernel {gpm.gp.kernel_}")

    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax[0, 0].plot(s1.t[:50], cov_true[:50])
    ax[1, 0].plot(s1.t[:50], cov_prior[0, :50])
    ax[2, 0].plot(s1.t[:50], cov_post[0, :50])

    cmap = 'viridis'
    if min(cov_true) < 0:
        cmap = "PiYG"
    im1 = ax[0, 1].imshow(cov_true_matrix[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    im2 = ax[1, 1].imshow(cov_prior[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    im3 = ax[2, 1].imshow(cov_post[:50, :50], cmap=cmap)
    fig.colorbar(im1)
    fig.colorbar(im2)
    fig.colorbar(im3)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].plot(ts1.t, mean_prior,  "b-", label="posterior")
    ax[0].fill_between(ts1.t, mean_prior-std_prior, mean_prior + std_prior,
                     color='b', alpha=0.2, label='CI')
    ax[0].plot(ts1.t, ts1.sum(), 'r-', label="true", alpha=0.5)

    ax[1].plot(x_train, y_train, 'yo', label="observations", markersize=12)
    ax[1].plot(ts1.t, mean_post,  "b-", label="posterior")
    ax[1].fill_between(ts1.t, mean_post-std_post, mean_post + std_post,
                     color='b', alpha=0.2, label='CI')
    ax[1].plot(ts1.t, ts1.sum(), 'r-', label="true", alpha=0.5)
    plt.legend()
    plt.show()

    return s1_config


if __name__ == "__main__":

    # Simulate AR(1) with phi = 0.8
    phi = 0.8
    wn_scale = 1

    ar = np.array([1, -phi])
    # s1_config = simulate_plot_ar(ar, wn_scale)

    ar = np.array([1, -phi, -phi/2])
    s2_config = simulate_plot_ar(ar, wn_scale)









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

