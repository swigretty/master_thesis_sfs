from dataclasses import dataclass, asdict

import pandas as pd
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared,\
    ConstantKernel, RationalQuadratic, Matern, Product, Sum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem
import numpy as np
import re
from functools import partial
from logging import getLogger

logger = getLogger(__name__)


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
    if y_std.ndim > 1:
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


class ARKernel(Matern):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), order=1):
        nu = order - 0.5
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.order = order


class GPR(GaussianProcessRegressor):
    """
    https://stackoverflow.com/questions/62376164/how-to-change-max-iter-in-optimize-function-used-by-sklearn-gaussian-process-reg
    """
    # def __init__(self, *args, max_iter=15000, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._max_iter = max_iter

    # def _constrained_optimization(self, obj_func, initial_theta, bounds):
    #     def new_optimizer(obj_func, initial_theta, bounds):
    #         return scipy.optimize.minimize(
    #             obj_func,
    #             initial_theta,
    #             method="L-BFGS-B",
    #             jac=True,
    #             bounds=bounds,
    #             max_iter=self._max_iter,
    #         )
    #     self.optimizer = new_optimizer
    #     return super()._constrained_optimization(obj_func, initial_theta, bounds)

    def __init__(self, kernel, rng=None, kernel_approx=False, **kwargs):

        if rng is None:
            rng = np.random.default_rng()
        self.kernel_orig = kernel
        self.rng = rng
        self.kernel_approx = kernel_approx
        self.kernel_approx_method = partial(Nystroem, n_components=200, random_state=1)
        if self.kernel_approx:
            self.kernel_approx = self.kernel_approx_method(kernel, n_components=300)
            kernel = None

        super().__init__(kernel=kernel, random_state=rng.integers(0, 10), **kwargs)

    @classmethod
    def decompose_additive_kernel(cls, k):
        if isinstance(k, Sum):
            k1_list = cls.decompose_additive_kernel(k.k1)
            k2_list = cls.decompose_additive_kernel(k.k2)
            k_list = k1_list + k2_list
        else:
            k_list = [k]
        return k_list

    def predict_mean_decomposed(self, X):

        decompose_dict = {}

        if not hasattr(self, "X_train_"):
            logger.warning("The model has not been fitted")
            return

        kernel = self.kernel_
        scale_kernel = ConstantKernel(constant_value=1, constant_value_bounds="fixed")

        if isinstance(self.kernel_, Product) and isinstance(self.kernel_.k1, ConstantKernel):
            scale_kernel = self.kernel_.k1
            kernel = self.kernel_.k2

        kernel_list = self.decompose_additive_kernel(kernel)

        for kernel in kernel_list:
            kernel = kernel * scale_kernel
            K_trans = kernel(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            decompose_dict[f"{kernel}"] = y_mean

        return decompose_dict

    def predict(self, x: np.ndarray, return_std=False, return_cov=False):
        if self.kernel_approx:
            x = self.kernel_approx.transform(x)
        return super().predict(x, return_std=return_std, return_cov=return_cov)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        if self.kernel_approx:
            train_x = self.kernel_approx.fit_transform(train_x)
        super().fit(train_x, train_y)

    def sample_y(self, x, n_samples=1):
        y_mean, y_cov = self.predict(x, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = self.rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                self.rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples, y_mean, y_cov

    def sample_from_prior(self, x, n_samples, mean_f=lambda x: 0):
        gps = self
        if hasattr(self, "X_train_"):  # has already been fitted
            gps = self.__class__(kernel=self.kernel)  # sample_y does only need kernel information
        mean_f_val = np.array([mean_f(val) for val in x])
        mean_f_val_full = np.vstack([mean_f_val for i in range(n_samples)]).T
        y_samples, y_mean, y_cov = gps.sample_y(x, n_samples)
        return y_samples + mean_f_val_full, y_mean + np.mean(mean_f_val_full, axis=1), y_cov

    def sample_from_posterior(self, x, n_samples):
        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            raise Exception("GP has not been fitted yet, cannot sample from posterior")
        return self.sample_y(x, n_samples)


@dataclass
class GPData():
    """
    gp1 = GPData(x=np.array([1]), y = np.array([2]))
    """
    x: np.array
    y: np.array
    y_mean: np.array = None
    y_cov: np.array = None

    def __post_init__(self):
        self.index = 0
        self.check_dimensions()

    def check_dimensions(self):
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        if self.y.ndim == 2:
            self.y = self.y.reshape(-1)
        if self.y_mean.ndim == 2:
            self.y_mean = self.y_mean.reshape(-1)

        assert self.x.shape[0] == self.y.shape[0] == self.y_mean.shape[0]
        assert self.y_cov.shape == (len(self.x), len(self.x))

    def __len__(self):
        return len(self.x)

    def to_df(self):
        df = pd.DataFrame({k: v for k, v in asdict(self).items() if k not in ["y_cov", "x"]})
        df["y_var"] = np.diag(self.y_cov)
        if self.x.shape[1] > 1:
            for i in self.x.shape[1]:
                df[f"x{i}"] = self.x[:, i]
        else:
            df["x"] = self.x.reshape(-1)
        return df

    @classmethod
    def from_df(cls, df):
        return cls(**df.to_dict())

    def get_field_item(self, field, idx):
        value = getattr(self, field)
        if value is None:
            return value
        if field == "y_cov":
            if isinstance(idx, int):
                return value[idx, idx]
            return np.array([[self.y_cov[io, ii] for ii in idx] for io in idx])
        return value[idx]

    def __getitem__(self, idx):
        return self.__class__(**{k: self.get_field_item(k, idx) for k in asdict(self).keys()})

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self):
            self.index = 0
            raise StopIteration
        self.index += 1
        return self[self.index - 1]
