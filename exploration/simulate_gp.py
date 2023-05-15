from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl

from exploration.gp import GPModel, plot_gpr_samples
from exploration.constants import PLOT_PATH
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


def simulate_bp_gp(kernel, global_mean=120):
    gpm = GPModel(kernel=kernel)
    # y_samples = gpm.sample_from_prior(x, global_mean=global_mean)
    fig, ax = plt.subplots()
    x, y_samples = plot_gpr_samples(gpr_model=gpm, n_samples=10, ax=ax, global_mean=global_mean)
    plt.show()
    logger.info("Finished")
    return x, y_samples


if __name__ == "__main__":
    setup_logging()
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    kernel = 1 * Matern(nu=0.5, length_scale=1)
    x, y = simulate_bp_gp(kernel=kernel, global_mean=120)

    kernel = kernel + ConstantKernel(constant_value=120, constant_value_bounds=(60, 300))
    gpm = GPModel(kernel=kernel)
    gpm.fit_model(x, y)



    #
    # fig, ax = plt.subplots()
    # plot_gpr_samples(gpr_model=gpm.gp, n_samples=10, ax=ax)
    # plt.show()
    logger.info("Finished")








