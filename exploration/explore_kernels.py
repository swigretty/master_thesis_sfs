from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern
from exploration.constants import PLOT_PATH
import matplotlib.pyplot as plt
import numpy as np
from log_setup import setup_logging
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl

logger = getLogger(__name__)


def plot_kernel(kernel, t=np.linspace(0, 20, 200), kernel_config_list=None, plot_file=None):

    mpl.style.use('seaborn-v0_8')
    if kernel_config_list is None:
        kernels = [kernel()]
    else:
        kernels = [kernel(**conf) for conf in kernel_config_list]

    plot_path = PLOT_PATH / "kernels"
    if plot_file is None:
        plot_file = plot_path / f"{kernel.__name__}_{len(kernels)}.pdf"
    else:
        plot_file = plot_path / plot_file
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    ax[0].set_title("Kernel Function")
    ax[1].set_title("Sample Path")

    for i, k in enumerate(kernels):
        # color = CSS4_COLORS[i * 3]
        t_max = k.length_scale * 5
        idx_max = next(idx for idx, x in enumerate(t) if x >= t_max)

        K_t = k(np.array(t).reshape(-1, 1))
        mu = np.zeros(len(t))
        ax[0].plot(t[: idx_max], K_t[0, : idx_max], label=f"{kernel_config_list[i]}")
        # cmap = 'viridis'
        # if np.min(K_t) < 0:
        #     cmap = "PiYG"
        # im1 = ax[i, 1].imshow(K_t, cmap=cmap)
        #
        # fig.colorbar(im1)

        ax[1].plot(t, np.transpose(np.random.multivariate_normal(mean=mu, cov=K_t, size=1)))
    ax[0].legend()
    fig.savefig(plot_file)
    plt.show()





if __name__ == "__main__":
    setup_logging()
    kernel = Matern

    plot_kernel(kernel, kernel_config_list=[{"nu": 0.5}, {"nu": 2.5}, {"nu": np.inf}])






