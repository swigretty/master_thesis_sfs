import numpy as np
from simulation.simulate_bp import simulate_bp_simple, random_cosinus_seasonal, cosinus_seasonal, get_design_matrix
from functools import partial
from exploration.blr import blr_corr, plot_blr_output, blr_simple
import matplotlib.pyplot as plt
from exploration.gp import fit_gp

def get_red_idx(n, data_fraction=0.3, weights=None):
    k = int(n * data_fraction)
    if weights is not None:
        weights = weights/sum(weights)
    return sorted(np.random.choice(range(n), size=k, replace=False, p=weights))


def simulate_random_seasonal(seas_ampl=10, ndays=4, samples_per_hour=10, data_fraction=0.3):
    seasonal_fun = partial(random_cosinus_seasonal, seas_ampl=seas_ampl)

    ts_true = simulate_bp_simple(seasonal_fun=seasonal_fun, ndays=ndays,
                                 samples_per_hour=samples_per_hour)
    fig = ts_true.plot()
    # plt.show(block=True)
    fig.savefig("ts_true_random_seasonal.svg")

    true_regression_line = ts_true.trend + cosinus_seasonal(ts_true.t, ts_true.period,
                                                                            seas_ampl=seas_ampl)

    return ts_true, true_regression_line


if __name__ == "__main__":
    data_fraction = 0.3

    simulation_config = dict(
        seas_ampl=10,
        ndays=4,
        samples_per_hour=10)

    ts_true, true_regression_line = simulate_random_seasonal(**simulation_config)

    red_idx = get_red_idx(len(ts_true.t), data_fraction=data_fraction)
    X = get_design_matrix(ts_true.t[red_idx], ts_true.period)
    y = ts_true.sum()[red_idx]

    idata = blr_simple(X, y)
    fig = plot_blr_output(idata, ts_true.t[red_idx], ts_true.t, true_regression_line)
    fig.savefig("blr_simple.svg")
    # idata = blr_corr(X, y)


