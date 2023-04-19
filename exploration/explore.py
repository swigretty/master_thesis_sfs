import numpy as np
from simulation.simulate_bp import simulate_bp_simple, random_cosinus_seasonal, cosinus_seasonal, get_design_matrix, linear_trend
from functools import partial
from exploration.blr import blr_corr, plot_blr_output, blr_simple
import matplotlib.pyplot as plt
from exploration.gp import fit_gp


def get_red_idx(n, data_fraction=0.3, weights=None):
    k = int(n * data_fraction)
    if weights is not None:
        weights = weights/sum(weights)
    return sorted(np.random.choice(range(n), size=k, replace=False, p=weights))


def simulate_cos_seasonal(seas_ampl=10, ndays=4, samples_per_hour=10):
    seasonal_fun = partial(cosinus_seasonal, seas_ampl=seas_ampl)

    ts_true = simulate_bp_simple(seasonal_fun=seasonal_fun, ndays=ndays,
                                 samples_per_hour=samples_per_hour)
    fig = ts_true.plot()
    # plt.show(block=True)
    fig.savefig("ts_true_cos_seasonal.svg")

    true_regression_line = ts_true.trend + cosinus_seasonal(ts_true.t, ts_true.period,
                                                                            seas_ampl=seas_ampl)

    return ts_true, true_regression_line


def simulate_random_seasonal(seas_ampl=10, ndays=4, samples_per_hour=10):
    seasonal_fun = partial(random_cosinus_seasonal, seas_ampl=seas_ampl)

    ts_true = simulate_bp_simple(seasonal_fun=seasonal_fun, ndays=ndays,
                                 samples_per_hour=samples_per_hour)
    fig = ts_true.plot()
    # plt.show(block=True)
    fig.savefig("ts_true_random_seasonal.svg")

    true_regression_line = ts_true.trend + cosinus_seasonal(ts_true.t, ts_true.period,
                                                                            seas_ampl=seas_ampl)

    return ts_true, true_regression_line


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


if __name__ == "__main__":
    data_fraction = 0.1

    simulation_config = dict(
        seas_ampl=2,
        ndays=4,
        samples_per_hour=10)

    # ts_true, true_regression_line = simulate_random_seasonal(**simulation_config)
    ts_true, true_regression_line = simulate_cos_seasonal(**simulation_config)

    red_idx = get_red_idx(len(ts_true.t), data_fraction=data_fraction,
                          weights=get_binary_weights(len(ts_true.t)))
    t = ts_true.t[red_idx]
    X = get_design_matrix(ts_true.t[red_idx], ts_true.period)
    y = ts_true.sum()[red_idx]

    gp_fitted, gp = fit_gp(t.reshape(-1, 1), y, period=ts_true.period)
    cov = np.dot(gp_fitted.L_, np.transpose(gp_fitted))
    cov = gp_fitted.kernel_

    gp_mean, gp_std = gp.predict(ts_true.t.reshape(-1, 1), return_std=True)
    fig, ax = plt.subplots()
    ax.plot(ts_true.t, gp_mean,  "b-", label="predicted")
    plt.fill_between(ts_true.t, gp_mean-gp_std, gp_mean + gp_std,
                     color='b', alpha=0.2, label='CI')
    ax.plot(t, y, 'y+', alpha=0.8, label="observations")

    ax.plot(ts_true.t, ts_true.trend + ts_true.seasonal + ts_true.resid, 'r-', label="true")
    plt.legend()
    plt.show()



    # idata = blr_simple(X, y)
    # fig = plot_blr_output(idata, ts_true.t[red_idx], ts_true.t, true_regression_line)
    # fig.savefig("blr_simple.svg")
    # idata = blr_corr(X, y)


