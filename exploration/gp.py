from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state


def plot_gpr_samples(gpr_model, n_samples, ax, global_mean=0):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.gp.predict(X, return_std=True)
    y_samples = gpr_model.sample_from_prior(X, n_samples, global_mean=global_mean)

    y_mean = np.zeros(y_samples.shape[0]) + global_mean

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
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
    return X, y_samples
    # ax.set_ylim([-3, 3])


class ARKernel(Matern):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), order=1):
        nu = order - 0.5
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.order = order


class GPModel(object):
    def __init__(self, kernel, rng=None, meas_noise=0, kernel_approx=False, normalize_y=True):
        if rng is None:
            rng = np.random.default_rng()

        self.kernel = kernel
        self.rng = rng
        self.kernel_approx_method = partial(Nystroem, n_components=200, random_state=1)

        self.gp = partial(GaussianProcessRegressor, n_restarts_optimizer=5, normalize_y=normalize_y,
                          random_state=0, alpha=meas_noise)

        if not kernel_approx:
            self.gp = self.gp(kernel=self.kernel)
            self.kernel_approx = None
        else:
            self.gp = self.gp()
            self.kernel_approx = self.kernel_approx_method(kernel=self.kernel, n_components=300)

    def predict(self, x: np.ndarray, return_std=False, return_cov=False):
        if self.kernel_approx is not None:
            x = self.kernel_approx.transform(x)
        gp_mean, gp_unc = self.gp.predict(x, return_std=return_std, return_cov=return_cov)
        return gp_mean, gp_unc

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        if self.kernel_approx is not None:
            train_x = self.kernel_approx.fit_transform(train_x)
        self.gp.fit(train_x, train_y)
        pass

    def sample_from_prior(self, x, n_samples, global_mean=0):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        y_cov = self.kernel(x)
        y_mean = np.zeros(x.shape[0]) + global_mean
        y_samples = self.rng.multivariate_normal(y_mean, y_cov, n_samples).T
        return y_samples


def fit_gp(X, y, period):
    # long_term_trend_kernel = 50.0 ** 2 * RBF(length_scale=50.0)

    k0 = WhiteKernel()

    k1 = RBF(length_scale=int(period/10)) * ExpSineSquared(length_scale=1.0, periodicity=period)

    k2 = RBF(length_scale=int(period/10))

    kernel = k0 + k2 + k1

    # FINAL_KERNEL = RBF(length_scale=1) * ExpSineSquared(length_scale=1, periodicity=1) + RBF(
    #     length_scale=1) * RationalQuadratic(alpha=1, length_scale=1) * DotProduct(sigma_0=1) + RBF(
    #     length_scale=1) * RationalQuadratic(alpha=1, length_scale=1) + RationalQuadratic(alpha=1, length_scale=1)

    gp1 = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=0.0
    )

    # # Plot prior
    # gp1_prior_samples = gp1.sample_y(X=X, n_samples=100)
    # fig, ax = plt.subplots()
    # ax.plot(X[:, 1], y, color="black", linestyle="dashed", label="Observations")
    # ax.plot(X[:, 1], gp1_prior_samples, color="tab:blue", alpha=0.4, label="prior samples")
    # plt.legend()

    gp_fitted = gp1.fit(X, y)

    return gp_fitted, gp1

