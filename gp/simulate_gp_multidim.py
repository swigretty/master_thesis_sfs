from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl

from gp.gp import GPR, plot_gpr_samples, plot_kernel_function
from constants.constants import OUTPUT_PATH
from exploration.explore import get_red_idx
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


def simulate_bp_gp(kernel, global_mean=120):
    gpm = GPR(kernel=kernel)
    # y_samples = gpm.sample_from_prior(x, global_mean=global_mean)
    x, y_samples = plot_gpr_samples(gpr_model=gpm, n_samples=10, ax=ax, global_mean=global_mean)
    plt.show()
    logger.info("Finished")
    return x, y_samples


def sim_fit_plot_gp(x=np.linspace(0, 20, 100), global_mean=120,
                    kernel=1 * Matern(nu=0.5, length_scale=1), data_fraction=0.3,
                    data_fraction_weights=None):
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    gpm = GPR(kernel=kernel, normalize_y=False)

    y_prior, y_prior_mean, y_prior_cov = gpm.sample_from_prior(x, n_samples=4, global_mean=global_mean)
    y_prior_std = np.diag(y_prior_cov)

    ylim = None
    if y_prior_std[-1] > 50:
        plot_lim = 50
        ylim = [global_mean - plot_lim, global_mean + plot_lim]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 10))

    plot_gpr_samples(ax[0, 0], x, y_prior, y_prior_mean, np.diag(y_prior_cov),
                     ylim=ylim)
    plot_kernel_function(ax[0, 1], x, kernel)

    # Posterior
    y = y_prior[:, 0]
    x = np.column_stack((np.ones((x.shape[0])), x))

    idx = get_red_idx(x.shape[0], data_fraction=data_fraction, weights=data_fraction_weights)

    y_red = y[idx]
    x_red = x[idx]

    # kernel = kernel + ConstantKernel(constant_value=global_mean)
    gpm = GPR(kernel=kernel, normalize_y=False)
    gpm.fit(x_red, y_red)
    y_post, y_post_mean, y_post_cov = gpm.sample_from_posterior(x, n_samples=4)
    plot_gpr_samples(ax[1, 0], x, y_post, y_post_mean, np.diag(y_post_cov))
    ax[1, 0].scatter(x_red[:, 1], y_red, color="red", zorder=10, label="Observations")
    ax[1, 0].set_title("Samples from posterior distribution")
    plot_kernel_function(ax[1, 1], x, gpm.gp.kernel_)
    # ax[1, 0].legend(loc="lower left")
    return fig


if __name__ == "__main__":
    setup_logging()
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    for data_fraction in np.linspace(0.1, 1, 6):
        kernel_ar1 = 1 * Matern(nu=0.5, length_scale=1)
        fig = sim_fit_plot_gp(data_fraction=data_fraction, global_mean=120, kernel=kernel_ar1)
        fig.savefig(OUTPUT_PATH / f"sim_fit_plot_const_{data_fraction:.2f}.pdf")

        kernel_dot = kernel_ar1 + DotProduct(sigma_0=1)
        fig = sim_fit_plot_gp(data_fraction=data_fraction, global_mean=120, kernel=kernel_dot)
        fig.savefig(OUTPUT_PATH / f"sim_fit_plot_dot_{data_fraction:.2f}.pdf")

        kernel_sin = kernel_ar1 + 10.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, periodicity_bounds=(2,4))
        fig = sim_fit_plot_gp(data_fraction=data_fraction, global_mean=120, kernel=kernel_sin)
        fig.savefig(OUTPUT_PATH / f"sim_fit_plot_sin_{data_fraction:.2f}.pdf")


    #
    # fig, ax = plt.subplots()
    # plot_gpr_samples(gpr_model=gpm.gp, n_samples=10, ax=ax)
    # plt.show()
    logger.info("Finished")








