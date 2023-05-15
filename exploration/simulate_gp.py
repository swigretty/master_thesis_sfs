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
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


def simulate_bp_gp(kernel, global_mean=120):
    gpm = GPModel(kernel=kernel)
    # y_samples = gpm.sample_from_prior(x, global_mean=global_mean)
    x, y_samples = plot_gpr_samples(gpr_model=gpm, n_samples=10, ax=ax, global_mean=global_mean)
    plt.show()
    logger.info("Finished")
    return x, y_samples


if __name__ == "__main__":
    setup_logging()
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    x = np.linspace(0, 5, 100)

    global_mean = 120
    kernel = 1 * Matern(nu=0.5, length_scale=1)
    gpm = GPModel(kernel=kernel)

    y_prior, y_prior_mean, y_prior_cov = gpm.sample_from_prior(x, n_samples=10, global_mean=global_mean)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plot_gpr_samples(ax[0, 0], x, y_prior, y_prior_mean, np.diag(y_prior_cov))
    plot_kernel_function(ax[0, 1], x, kernel)

    # Posterior
    y = y_prior
    kernel = kernel + ConstantKernel(constant_value=global_mean)
    gpm = GPModel(kernel=kernel)
    gpm.fit_model(x, y)
    y_post, y_post_mean, y_post_cov = gpm.sample_from_posterior(x, n_samples=10)
    plot_gpr_samples(ax[1, 0], x, y_post, y_post_mean, np.diag(y_post_cov))
    ax[1, 0].scatter(x, y, color="red", zorder=10, label="Observations")
    ax[1, 0].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    ax[1, 0].set_title("Samples from posterior distribution")



    #
    # fig, ax = plt.subplots()
    # plot_gpr_samples(gpr_model=gpm.gp, n_samples=10, ax=ax)
    # plt.show()
    logger.info("Finished")








