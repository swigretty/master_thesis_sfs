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

logger = getLogger(__name__)


def plot_kernels(kernels, t=np.linspace(0, 20, 200), plot_file=None):

    mpl.style.use('seaborn-v0_8')
    first_kernel = kernels[0]

    plot_path = OUTPUT_PATH
    if plot_file is None:
        plot_file = plot_path / f"{first_kernel.__class__.__name__}_{len(kernels)}.pdf"
    else:
        plot_file = plot_path / plot_file

    plot_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    ax[0].set_title("Kernel Function")
    ax[1].set_title("Sample Path")

    try:
        max_length_scale = max([k.length_scale for k in kernels])
        t_max = max_length_scale * 5
    except Exception:
        t_max = max(t)

    for i, k in enumerate(kernels):
        # color = CSS4_COLORS[i * 3]
        try:
            idx_max = next(idx for idx, x in enumerate(t) if x >= t_max)
        except StopIteration:
            idx_max = len(t)
        K_t = k(np.array(t).reshape(-1, 1))
        mu = np.zeros(len(t))
        ax[0].plot(t[: idx_max], K_t[0, : idx_max], label=f"{k}")
        # cmap = 'viridis'
        # if np.min(K_t) < 0:
        #     cmap = "PiYG"
        # im1 = ax[i, 1].imshow(K_t, cmap=cmap)
        #
        # fig.colorbar(im1)
        y_sim = np.transpose(np.random.multivariate_normal(mean=mu, cov=K_t, size=1))
        ax[1].plot(t, y_sim, label=f"{np.var(y_sim)}")
    ax[0].legend()
    ax[1].legend()

    fig.savefig(plot_file)


if __name__ == "__main__":
    setup_logging()
    days = 3
    h_per_day = 24
    samples_per_hour = 20
    t = np.linspace(0, days * h_per_day, days * h_per_day * samples_per_hour)

    kernels = [Matern(nu=nu) for nu in [0.5, 2.5, np.inf]]
    plot_kernels(kernels)
    #
    kernels = [ExpSineSquared(length_scale=sc, periodicity=h_per_day) for sc in [0.1, 1, 10]]
    plot_kernels(kernels, plot_file="sin_len.pdf", t=t)
    #
    kernels = [ExpSineSquared(length_scale=1, periodicity=h_per_day) * RBF(length_scale=ls) for ls in [0.1, 1, 10,
                                                                                                        100]]
    plot_kernels(kernels, plot_file="sinrbf_len.pdf", t=t)
    #
    kernels = [c * RBF(length_scale=1) for c in [0.1, 1, 10, 100]]
    plot_kernels(kernels, plot_file="RBF_scale.pdf", t=t)

    kernels = [c * Matern(length_scale=1, nu=0.5) for c in [0.1, 1, 10, 100]]
    plot_kernels(kernels, plot_file="OU_scale.pdf", t=t)

    kernels = [c * ExpSineSquared(length_scale=3, periodicity=h_per_day) for c in [0.1, 1, 10, 100]]
    plot_kernels(kernels, plot_file="sin_scale.pdf", t=t)
