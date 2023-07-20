import re
import functools
import matplotlib.pyplot as plt
from logging import getLogger
from constants.constants import OUTPUT_PATH

logger = getLogger(__name__)


# TODO combine Plotter and ts_plotter
class Plotter():
    def __init__(self, f):
        self.func = f

    def __call__(self, instance, *args, figname_suffix="", ax=None, **kwargs):
        fig = None
        output_path = getattr(instance, "output_path", None)

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        value = self.func(instance, *args, ax=ax, **kwargs)

        if figname_suffix and not figname_suffix.startswith("_"):
            figname_suffix = f"_{figname_suffix}"

        if fig is not None:
            fig.tight_layout()
            if output_path:
                fig.savefig(output_path / f"{self.func.__name__}{figname_suffix}.pdf")
                plt.close()
        return value

    def __get__(self, instance, owner):
        from functools import partial
        return partial(self.__call__, instance)


def ts_plotter(figname_suffix="", output_path=OUTPUT_PATH):
    def ts_plotter_inner(func):
        """ Function either plots on subplot of existing figure (ax is defined in kwargs)
         or creates new figure with 1 plot (ax is None) and stores the figure in output_path"""
        @functools.wraps(func)
        def wrapper_plotter(*args, ax=None, **kwargs):
            nonlocal figname_suffix
            nonlocal output_path

            fig = None

            if ax and figname_suffix:
                logger.warning(f"{figname_suffix=} will be ignored since figure not store in function: {func.__name__}")
            if ax is None:
                fig, ax = plt.subplots(nrows=1, ncols=1, figisize=(10, 6))

            value = func(*args, ax=ax, **kwargs)

            if figname_suffix and not figname_suffix.startswith("_"):
                figname_suffix = f"_{figname_suffix}"

            if fig is not None:
                fig.tight_layout()
                fig.savefig(output_path / f"{func.__name__}{figname_suffix}.pdf")
                plt.close()
            return value

        return wrapper_plotter
    return ts_plotter_inner


@ts_plotter()
def plot_kernel_function(x, kernel, ax=None):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    KXX = kernel(x)
    k_values = KXX[0, :]
    # idx_max = np.argmax(k_values < 0.01)
    # ax.plot(x[:idx_max], KXX[0, :idx_max])

    if x.shape[1] > 1:
        x = x[:, 1]

    ax.plot(x, KXX[0, :])

    title = re.sub("(.{120})", "\\1\n", str(kernel), 0, re.DOTALL)
    ax.set_title(title)


@ts_plotter()
def plot_posterior(x, y_post_mean, y_post_std=None, x_red=None, y_red=None,
                   y_true=None, ax=None):
    plot_gpr_samples(x, y_post_mean, y_post_std, y=None, ax=ax)
    if y_true is not None:
        ax.plot(x, y_true, "r:")
    if (x_red is not None) and (y_red is not None):
        ax.scatter(x_red, y_red, color="red", zorder=5, label="Observations")
    ax.set_title("Samples from posterior distribution")


@ts_plotter()
def plot_gpr_samples(x, y_mean, y_std=None, y=None, ylim=None, ax=None):
    """
    y has shape (n_prior_samples, n_samples_per_prior)
    """

    if x.ndim > 1:
        if x.shape[1] > 1:
            x = x[:, 1]
        else:
            x = x.reshape(-1)
    if y_std is not None and y_std.ndim > 1:
        raise ValueError(f"y_std must have 1 dimension not {y_std.ndim}")
    if y is not None:
        for idx, single_prior in enumerate(y):
            ax.plot(
                x,
                single_prior,
                linestyle="--",
                alpha=0.7,
                label=f"Sampled function #{idx + 1}",
            )

    ax.plot(x, y_mean, color="black", label="Mean")
    if y_std is not None:
        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
            label=r"$\pm$ 1 std. dev.",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if ylim:
        ax.set_ylim(ylim)
