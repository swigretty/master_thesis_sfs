from simulation.simulate_bp import simulate_bp_simple, random_cosinus_seasonal, cosinus_seasonal
from functools import partial
import pymc3 as pm
from pymc3 import HalfCauchy, Model, Normal, glm, plot_posterior_predictive_glm, sample
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import numpy as np
import random


def get_design_matrix(t, period):
    trad = t*2*np.pi*(1/period)
    return np.column_stack(([1]*len(t), t, np.cos(trad), np.sin(trad)))


def blr_simple(X, y):
    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)

        mu = sum([Normal("b"+str(i), 0, sigma=20) * X[:, i] for i in range(X.shape[1])])

        # Define likelihood
        likelihood = Normal("y", mu=mu, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trace = sample(3000, return_inferencedata=True)
        trace.posterior["y_model"] = sum([trace.posterior["b"+str(i)] * xr.DataArray(X[:, i]) for i in
                                          range(X.shape[1])])
    return trace

def plot_blr_output(trace, x, x_true, true_regression_line):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_true, true_regression_line, label="true regression line", lw=3.0, c="y")
    az.plot_lm(idata=trace, x=x, y="y", num_samples=100, axes=ax, y_model="y_model")
    ax.set_title("Posterior predictive regression lines")
    ax.set_xlabel("t")
    ax.set_title("Posterior predictive regression lines")
    return fig


def blr_corr(X, y):
    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors

        mu = sum([Normal("b"+str(i), 0, sigma=20) * X[:, i] for i in range(X.shape[1])])
        sd_dist = pm.HalfCauchy.dist(beta=5)
        # sd_dist = pm.Exponential.dist(1.0, shape=2)
        # chol, corr, stds = pm.LKJCholeskyCov(
        #     "chol", n=len(y), eta=2, sd_dist=sd_dist, compute_corr=True)
        L_packed = pm.LKJCholeskyCov(
            "chol", n=len(y), eta=2, sd_dist=sd_dist)
        L = pm.expand_packed_triangular(len(y), L_packed)
        likelihood = pm.MvNormal('y', mu=mu, chol=L, observed=y)
        trace = sample(300, return_inferencedata=True)
        trace.posterior["y_model"] = sum(
            [trace.posterior["b" + str(i)] * xr.DataArray(X[:, i]) for i in range(X.shape[1])])

    return trace


if __name__ == "__main__":
    seas_ampl = 10
    ndays = 4
    samples_per_hour = 10
    data_fraction = 0.3

    seasonal_fun = partial(random_cosinus_seasonal, seas_ampl=seas_ampl)
    ts_true = simulate_bp_simple(seasonal_fun=seasonal_fun, ndays=ndays, samples_per_hour=samples_per_hour)
    fig = ts_true.plot()
    # plt.show(block=True)
    fig.savefig("plot.svg")
    X = get_design_matrix(ts_true.t, ts_true.period)
    y = ts_true.sum()
    red_idx = sorted(random.sample(range(len(y)), int(data_fraction*len(y))))
    X_red = X[red_idx, :]
    y_red = y[red_idx]

    idata = blr_corr(X_red, y_red)
    true_regression_line = cosinus_seasonal(ts_true.t, ts_true.period, seas_ampl=seas_ampl)
    fig = plot_blr_output(idata, ts_true.t[red_idx], ts_true.t, true_regression_line)
    fig.savefig("blr_corr.svg")
