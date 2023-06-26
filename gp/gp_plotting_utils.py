import re


def plot_kernel_function(ax, x, kernel):
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


def plot_posterior(ax, x, y_post_mean, y_post_std=None, x_red=None, y_red=None,
                   y_true=None):
    plot_gpr_samples(ax, x, y_post_mean, y_post_std, y=None)
    if y_true is not None:
        ax.plot(x, y_true, "r:")
    if (x_red is not None) and (y_red is not None):
        ax.scatter(x_red, y_red, color="red", zorder=5, label="Observations")
    ax.set_title("Samples from posterior distribution")


def plot_gpr_samples(ax, x, y_mean, y_std=None, y=None, ylim=None):
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
