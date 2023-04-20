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
        "ndays": 4,
        "samples_per_hour": 10
    }
    return default_config


if __name__ == "__main__":

    # Simulate AR(1) with phi = 0.8
    phi = 0.8
    wn_scale = 1
    s1_config = {"ar": np.array([1, -phi]), "arma_scale": wn_scale,  "ma": np.array([1])}
    s1 = BPTimseSeriesSimulator(**s1_config, **get_default_config())
    ts1 = s1.generate_sample()
    ts1.plot()
    plt.show()

    var_true = wn_scale/(1-phi**2)
    cov_true = arma_acovf(ar=s1_config["ar"], ma=s1_config["ma"], sigma2=s1_config["arma_scale"], nobs=s1.nsample)
    print(f"{cov_true[0]=}, {var_true=}")
    # arima = ARIMA(ts1.sum(), order=(1, 0, 0), trend_offset=0)
    # s1fit = arima.fit(gls=False)
    # ar_est = np.array([1, -s1fit.params[1]])
    # cov_est = arma_acovf(ar=ar_est, ma=s1_config["ma"], sigma2=s1fit.params[2], nobs=s1.nsample)

    cov_true_matrix = np.zeros(shape=(s1.nsample, s1.nsample))
    for (i, j) in itertools.product(range(s1.nsample), range(s1.nsample)):
        cov_true_matrix[i, j] = cov_true[i-j]
        cov_true_matrix[j, i] = cov_true[i-j]
    cor_true_matrix = cov_true_matrix/cov_true[0]

    # gpm = GPModel(kernel=cov_true[0] * get_AR_kernel(order=1, length_scale=5), normalize_y=False)
    # # sample from prior
    # mean_prior, cov_prior = gpm.predict(s1.t.reshape(-1, 1), return_cov=True)
    cov_prior = (cov_true[0] * get_AR_kernel(order=1, length_scale=5))(s1.t.reshape(-1, 1))

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(s1.t[:50], cov_true[:50])
    ax[1, 0].plot(s1.t[:50], cov_prior[0, :50])
    cmap = 'viridis'
    if min(cov_true) < 0:
        cmap = "PiYG"
    im1 = ax[0, 1].imshow(cov_true_matrix[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    im2 = ax[1, 1].imshow(cov_prior[:50, :50], cmap=cmap, vmax=max(cov_true), vmin=min(cov_true))
    fig.colorbar(im1)
    plt.show()









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

