from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl

from exploration.gp import GPModel, plot_gpr_samples, plot_kernel_function
from exploration.constants import PLOT_PATH
from exploration.explore import get_red_idx
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


def sim_fit_plot_gp(x=np.linspace(0, 40, 200), kernel=1 * Matern(nu=0.5, length_scale=1),
                    mean_f=lambda x: 120, data_fraction_weights=None, meas_noise=0, figname=None):

    x, y_prior, y_prior_mean, y_prior_cov = sim_gp(x, kernel, mean_f)
    y_prior_noisy = y_prior
    if meas_noise:
        y_prior_noisy = y_prior + meas_noise * np.random.standard_normal((y_prior.shape))
        y_prior_cov[np.diag_indices_from(y_prior_cov)] += meas_noise

    fig_list = []
    for data_fraction in np.linspace(0.1, 1, 4):
        fig = fit_plot_gp(x, y_prior_noisy, y_prior_mean, y_prior_cov, kernel=kernel,
                          data_fraction=data_fraction, data_fraction_weights=data_fraction_weights,
                          meas_noise=meas_noise, y_true=y_prior)
        if figname is not None:
            fig.savefig(PLOT_PATH / f"{figname}_{data_fraction:.2f}.pdf")
        fig_list.append(fig)
    return fig


def sim_gp(x, kernel, mean_f):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    gpm = GPModel(kernel=kernel, normalize_y=False)
    y_prior, y_prior_mean, y_prior_cov = gpm.sample_from_prior(
        x, n_samples=5, mean_f=mean_f)
    return x, y_prior, y_prior_mean, y_prior_cov


def fit_plot_gp(x, y_prior, y_prior_mean, y_prior_cov,
                kernel=1 * Matern(nu=0.5, length_scale=1),
                data_fraction=0.3,
                data_fraction_weights=None,
                meas_noise=0.1, y_true=None):

    y_prior_std = np.diag(y_prior_cov)
    ylim = None
    if any(y_prior_std) > 50:
        plot_lim = 50
        ylim = [np.mean(y_prior) - plot_lim, np.mean(y_prior) + plot_lim]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 15))

    plot_gpr_samples(ax[0, 0], x, y_prior_mean, y_prior_std, y=y_prior, ylim=ylim)
    plot_kernel_function(ax[0, 1], x, kernel)
    ax[0, 0].set_title("Samples from prior distribution")

    # Posterior
    # x = np.column_stack((np.ones((x.shape[0])), x))
    sample_idx = 0
    idx = get_red_idx(x.shape[0], data_fraction=data_fraction, weights=data_fraction_weights)
    y_red = y_prior[idx, sample_idx]
    x_red = x[idx]
    kernel = kernel + ConstantKernel(constant_value=np.mean(y_prior))
    gpm = GPModel(kernel=kernel, normalize_y=False, meas_noise=meas_noise)
    gpm.fit_model(x_red, y_red)
    y_post, y_post_mean, y_post_cov = gpm.sample_from_posterior(x, n_samples=1)
    plot_gpr_samples(ax[1, 0], x, y_post_mean, np.diag(y_post_cov), y=None)
    if y_true is not None:
        ax[1, 0].plot(x, y_true[:, sample_idx], "r:")
    ax[1, 0].scatter(x_red, y_red, color="red", zorder=5, label="Observations")
    ax[1, 0].set_title("Samples from posterior distribution")
    plot_kernel_function(ax[1, 1], x, gpm.gp.kernel_)
    # ax[1, 0].legend(loc="lower left")
    return fig


def mean_fun_const(x):
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    return 120


if __name__ == "__main__":
    setup_logging()

    # measuring time in hours
    n_days = 3
    samples_per_hour = 10
    period_day = 24
    period_week = 7 * period_day
    x = np.linspace(0, period_day * n_days, period_day * n_days * samples_per_hour)

    # Simple Kernels
    ar1_kernel = 1 * Matern(nu=0.5, length_scale=3, length_scale_bounds="fixed")
    long_term_trend_kernel = 4 * RBF(length_scale=50, length_scale_bounds="fixed")
    short_term_trend_kernel = 4 * RBF(length_scale=3, length_scale_bounds="fixed")
    short_cycle_kernel = 10 * ExpSineSquared(length_scale=3, periodicity=period_day, periodicity_bounds="fixed",
                                             length_scale_bounds="fixed")
    long_cycle_kernel = 10 * ExpSineSquared(length_scale=3, periodicity=period_week, periodicity_bounds="fixed",
                                            length_scale_bounds="fixed")
    kernel_dot = DotProduct(sigma_0=1, sigma_0_bounds="fixed")

    kernels = {"ar1": ar1_kernel, "dot": ar1_kernel + kernel_dot, "rbf": ar1_kernel + long_term_trend_kernel,
               "sin": ar1_kernel + short_cycle_kernel,
               "sinrbf": ar1_kernel + short_cycle_kernel * long_term_trend_kernel}

    base_config = dict(
        x=x,
        meas_noise=0,
        mean_f=mean_fun_const
    )

    for k_name, k in kernels.items():
        sim_fit_plot_gp(kernel=k, figname=f"sim_fit_gp_{k_name}", **base_config)









