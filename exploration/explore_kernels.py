from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern
from pathlib import Path
from constants.constants import OUTPUT_PATH
import matplotlib.pyplot as plt
import numpy as np
from log_setup import setup_logging
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl
import pandas as pd

logger = getLogger(__name__)


def get_info(sample):
    return {"var": np.var(sample), "max-min": np.max(sample) - np.min(sample)}


def plot_sample_path(t, kernel, ax=None, nsim=1):
    mu = np.zeros(len(t))
    K = kernel(np.array(t).reshape(-1, 1))
    sim_info = []
    y_sim = np.random.multivariate_normal(mean=mu, cov=K, size=nsim)

    plot_sample = y_sim[0]
    if ax is not None:
        ax.set_title("Sample Path")
        ax.plot(t, plot_sample, label=f"var: {get_info(plot_sample)['var']:.2f}, "
                                      f"max-min: {get_info(plot_sample)['max-min']}")

    for sample in y_sim:
        sim_info.append(get_info(sample))

    sim_info_mean = pd.DataFrame(sim_info).mean().to_dict()
    sim_info_mean = {k: np.round(v, decimals=4) for k, v in sim_info_mean.items()}

    logger.info(f"{kernel}: {sim_info_mean}")
    return sim_info_mean


def plot_kernel(t, kernel, ax):
    ax.set_title("Kernel Function")

    K = kernel(np.array(t).reshape(-1, 1))
    k_1d = K[0, :]
    ax.plot(t, k_1d, label=f"{kernel}")
    # logger.info(kernel)
    # logger.info(f"Kmax-Kmin: {np.max(K) - np.min(K)}")


def plot_kernels(kernels, t=np.linspace(0, 20, 200), plot_file=None, mode_values=None, mode_name="mode", nsim=1):
    nrows = 1
    ncols = 2
    if mode_values is not None:
        ncols += 1
    mpl.style.use('seaborn-v0_8')
    first_kernel = kernels[0]
    if hasattr(first_kernel, "k1"):
        first_kernel = first_kernel.k1
    plot_path = OUTPUT_PATH
    plot_path.mkdir(parents=True, exist_ok=True)
    if plot_file is None:
        plot_file = f"{first_kernel.__class__.__name__}"
    plot_file = f"{plot_file}_{mode_name}.pdf"

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12*ncols, 8*nrows))
    try:
        max_length_scale = max([k.length_scale for k in kernels])
        t_max = max_length_scale * 5
    except Exception:
        t_max = max(t)

    try:
        idx_max = next(idx for idx, x in enumerate(t) if x >= t_max)
    except StopIteration:
        idx_max = len(t)

    for i, k in enumerate(kernels):
        # color = CSS4_COLORS[i * 3]
        plot_kernel(t[:idx_max], k, ax[0])
        plot_sample_path(t, k, ax=ax[1], nsim=nsim)

    if mode_values is not None:
        plot_sim_info(kernels, mode_values=mode_values, mode_name=mode_name, t=t, ax=ax[2])

    [a.legend() for a in ax]

    fig.savefig(plot_path / plot_file)


def plot_sim_info(kernels, mode_values, mode_name="mode", t=np.linspace(0, 20, 200), nsim=1000, ax=None):

    sim_info = []
    for i, k in enumerate(kernels):
        sim_info.append(plot_sample_path(t, k, nsim=nsim))
    sim_info_df = pd.DataFrame(sim_info)
    sim_info_df[mode_name] = mode_values
    if ax is not None:
        ax.plot(sim_info_df[mode_name], sim_info_df["var"], label="var")
        ax.plot(sim_info_df[mode_name], sim_info_df["max-min"], label="max-min")
        ax.set_xlabel(mode_name)

    return sim_info_df


if __name__ == "__main__":
    setup_logging()
    days = 3
    h_per_day = 24
    samples_per_hour = 20
    t = np.linspace(0, days * h_per_day, days * h_per_day * samples_per_hour)
    var = [1, 10, 100, 200, 500, 800, 1000]
    var = [5, 62, 14**2]
    length_scale = [1, 10, 100]

    # kernels = [Matern(nu=nu) for nu in [0.5, 2.5, np.inf]]
    # plot_kernels(kernels)
    # #
    # kernels = [ExpSineSquared(length_scale=sc, periodicity=h_per_day) for sc in [0.1, 1, 10]]
    # plot_kernels(kernels, plot_file="sin_len.pdf", t=t)
    # #
    # kernels = [ExpSineSquared(length_scale=1, periodicity=h_per_day) * RBF(length_scale=ls) for ls in [0.1, 1, 10,
    #                                                                                                     100]]
    # plot_kernels(kernels, plot_file="sinrbf_len.pdf", t=t)
    # #
    # kernels = [c * RBF(length_scale=1) for c in [0.1, 1, 10, 100]]
    # plot_kernels(kernels, plot_file="RBF_scale.pdf", t=t)
    #
    # kernels = [RBF(length_scale=ls) for ls in length_scale]
    # plot_kernels(kernels, t=t, mode_name="length_scale", mode_values=length_scale)
    #
    # kernels = [RBF(length_scale=50) * c for c in var]
    # plot_kernels(kernels, t=t, mode_name="rbf50_var", mode_values=var)
    # #
    # kernels = [Matern(length_scale=3, nu=0.5) * c for c in var]
    # plot_kernels(kernels, mode_name="rbf50_var", t=t, mode_values=var)

    kernels = [ExpSineSquared(length_scale=3, periodicity=h_per_day) * c for c in var]
    plot_kernels(kernels, t=t, mode_name="sin3_var", mode_values=var, nsim=100)

    # kernels = [WhiteKernel(noise_level=c)for c in var]
    # plot_kernels(kernels, t=t, mode_name="white_var", mode_values=var)
    #
    # kernels = [ExpSineSquared(length_scale=3, periodicity=h_per_day) * 14**2 + Matern(
    #     length_scale=1, nu=0.5) * 5 + RBF(length_scale=50) * 5 + WhiteKernel(noise_level=c) for c in var]
    # plot_kernels(kernels, t=t, mode_name="sin_rbf_ou_white", mode_values=var)




