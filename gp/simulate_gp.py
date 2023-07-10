import datetime
import inspect
from functools import cached_property
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
import logging
import scipy
import matplotlib as mpl
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from gp.gp_regressor import GPR
from gp.gp_plotting_utils import plot_kernel_function, plot_posterior, plot_gpr_samples, Plotter
from gp.gp_data import GPData
from constants.constants import OUTPUT_PATH
from exploration.explore import get_red_idx
from gp.simulate_gp_config import base_config, OU_KERNELS, PARAM_NAMES
from gp.evaluate import GPEvaluator
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


def weights_func_value(y):
    return y - np.min(y)


class GPSimulator():

    def __init__(self, x=np.linspace(0, 40, 200), kernel_sim=1 * Matern(nu=0.5, length_scale=1), mean_f=lambda x: 120,
                 meas_noise_var=0, kernel_fit=None, normalize_y=False, output_path=OUTPUT_PATH,
                 data_fraction_weights=None, data_fraction=0.3, f_true=None, meas_noise=None,
                 rng=None, normalize_kernel=False,
                 session_name=None):

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.x = x
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.mean_f = mean_f
        self.meas_noise_var = meas_noise_var
        self.offset = np.mean([mean_f(xi) for xi in self.x])

        self.data_fraction_weights = data_fraction_weights
        self.data_fraction = data_fraction

        self.kernel_sim = kernel_sim
        self.normalize_kernel = normalize_kernel
        if normalize_kernel:
            self.kernel_sim, self.meas_noise_var = self.get_normalized_kernel(kernel_sim,
                                                                              meas_noise_var=self.meas_noise_var)

        if kernel_fit is None:
            kernel_fit = self.kernel_sim

        self.kernel_fit = kernel_fit
        self.normalize_y = normalize_y
        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None, rng=rng,
                           alpha=0)
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=self.normalize_y, alpha=self.meas_noise_var, rng=rng)

        self.f_true = f_true
        self.meas_noise = meas_noise

        self.session_name = session_name
        if self.session_name is None:
            self.session_name = self.kernel_sim.__class__.__name__
        self.figname_suffix = f"{self.session_name}_mn{self.meas_noise_var:.2f}_df{self.data_fraction:.2f}"

        logger.info(f"Initialized {self.__class__.__name__} with \n {kernel_sim=} \n {kernel_fit=}")

    # def sim_gp(self, n_samples=5):
    #     # samples with measurement noise if predict_y=True
    #     y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
    #         self.x, n_samples=n_samples, predict_y=True)  # mean_f=self.mean_f
    #     data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov)
    #             for idx in range(n_samples)]
    #
    #     return data

    def sim_gp(self, n_samples=5, sim_y=True):
        # samples without measurement noise
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
            self.x, n_samples=n_samples)  # mean_f=self.mean_f
        # TODO should this be scaled as well ?
        # y_prior = y_prior + self.meas_noise * np.std(y_prior, axis=0) * np.random.standard_normal((y_prior.shape))
        y_prior_noisy = y_prior + self.meas_noise_var * self.rng.standard_normal((y_prior.shape))
        y_prior_cov_noisy = y_prior_cov + np.diag(np.repeat(self.meas_noise_var, len(self.x)))

        if sim_y:
            data = [GPData(x=self.x, y=y_prior_noisy[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov_noisy) for idx in
                    range(n_samples)]
        else:
            data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov) for idx in
                    range(n_samples)]

        return data

    def data_fraction_weights_seasonal(self):
        self.gpm_sim.fit(self.y_true.x, self.y_true.y)
        true_dec = self.gpm_sim.predict_mean_decomposed(self.x)
        fun = [fun for name, fun in true_dec.items() if "ExpSineSquared" in name]
        assert len(fun) == 1, "cannot extract seasonal pattern"
        weights = fun[0] - min(fun[0])
        return weights * 0.1

    @cached_property
    def f_true_post(self):
        self.gpm_sim.fit(self.f_true.x, self.f_true.y)
        f_post_mean, f_post_cov = self.gpm_sim.predict(self.y_true.x, return_cov=True)
        assert np.mean((self.f_true.y - f_post_mean) ** 2) < 10 ** (-4)
        return GPData(x=self.f_true.x, y_mean=f_post_mean, y_cov=f_post_cov)

    @property
    def meas_noise(self):
        return self._meas_noise

    @meas_noise.setter
    def meas_noise(self, data=None):
        self._meas_noise = data
        if data is None:
            self._meas_noise = self.meas_noise_var * self.rng.standard_normal(len(self.x))

    @property
    def y_true(self):
        """
        This represents the true (potentially noisy) and complete measurements
        """
        y_prior_noisy = self.f_true.y + self.meas_noise

        y_prior_cov_noisy = self.f_true.y_cov + np.diag(np.repeat(self.meas_noise_var, len(self.x)))

        return GPData(x=self.f_true.x, y=y_prior_noisy, y_cov=y_prior_cov_noisy)

    @property
    def f_true(self):
        """
        This represents the true and complete bp time series
        """
        return self._f_true

    @f_true.setter
    def f_true(self, data=None):
        self._f_true = data
        if data is None:
            self._f_true = self.sim_gp(n_samples=1, sim_y=False)[0]

    @cached_property
    def train_idx(self):
        if callable(self.data_fraction_weights):
            weights = self.data_fraction_weights(self.y_true)
        elif self.data_fraction_weights == "seasonal":
            weights = self.data_fraction_weights_seasonal()
        else:
            weights = self.data_fraction_weights

        idx = get_red_idx(len(self.x), data_fraction=self.data_fraction, weights=weights,
                          rng=self.rng)
        return idx

    @property
    def y_true_train(self):
        """
        This represents the (potentially noisy) subsampled measurements, used to fit the GP
        """
        return self.y_true[self.train_idx]

    @property
    def y_true_samples(self):
        if not hasattr(self, "_data_prior"):
            self.y_true_samples = self.sim_gp(n_samples=5)
        return self._data_prior

    @y_true_samples.setter
    def y_true_samples(self, data: list):
        data = data + [self.y_true]
        self._data_prior = data

    @property
    def param_sim(self):
        return self.extract_params_from_kernel(self.kernel_sim)

    @property
    def param_fit(self):
        if not hasattr(self.gpm_fit, "X_train_"):
            raise "GP has not been fitted yet"
        return self.extract_params_from_kernel(self.gpm_fit.kernel_)

    @cached_property
    def f_post(self):
        # f_post_cov is the predicted covariance matrix for f(x) (not for y(x))
        f_post_mean, f_post_cov = self.gpm_fit.predict(self.x, return_cov=True)
        return GPData(x=self.x, y_mean=f_post_mean, y_cov=f_post_cov)

    @property
    def data_mean(self):
        """
        Using the overall mean as prediction.
        cov is calculated assuming iid data.
        """
        sigma_mean = 1 / len(self.y_true_train.y) * np.var(self.y_true_train.y)
        y_cov = np.zeros((len(self.x), len(self.x)), float)
        np.fill_diagonal(y_cov, sigma_mean)
        y = np.repeat(np.mean(self.y_true_train.y), len(self.x))
        return GPData(x=self.x, y_mean=y, y_cov=y_cov)

    @property
    def test_idx(self):
        """
        or
        x = np.setdiff1d(self.data_post.x, self.data.x, assume_unique=True)
        return np.arange(len(self.x))[np.isin(self.x, x)]

        """
        return [idx for idx in range(len(self.x)) if self.x[idx] not in
                self.y_true_train.x]

    @staticmethod
    def subsample_data(data: GPData, data_fraction: float, data_fraction_weights=None, rng=None):
        if callable(data_fraction_weights):
            weights = data_fraction_weights(data.y)
        else:
            weights = data_fraction_weights
        idx = get_red_idx(len(data), data_fraction=data_fraction, weights=weights, rng=rng)
        return data[idx]

    @staticmethod
    def extract_params_from_kernel(kernel):
        return {k: v for k, v in kernel.get_params().items() if ("__" in k) and ("bounds" not in k) and any(
            [pn in k for pn in PARAM_NAMES])}

    @classmethod
    def get_normalized_kernel(cls, kernel, meas_noise_var=0):
        previousloglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)

        std_range = (0.95, 1.05)
        i = 0
        scale = 1
        y_std = 2

        while (std_range[0] > y_std) or (
                std_range[1] < y_std):

            kernel_ = ConstantKernel(constant_value=1/scale**2, constant_value_bounds="fixed") * kernel
            gps = cls(kernel_sim=kernel_, meas_noise_var=meas_noise_var / scale ** 2, rng=np.random.default_rng(11))
            data_sim = gps.sim_gp(n_samples=100)
            y_std = np.mean([np.std(d.y) for d in data_sim])
            scale *= y_std
            i += 1
            if i > 20:
                break

        logger.setLevel(previousloglevel)

        logger.info(f"final kernel {kernel_} with scaling {1/scale} and {y_std=}")

        return kernel_, 1 / scale * meas_noise_var

    def choose_sample_from_prior(self, data_index: int = 0):
        data_single = copy(self.y_true_samples)
        data_single.y = self.y_true_samples.y[:, data_index]
        return data_single

    def fit(self, refit=False):
        if (not hasattr(self.gpm_fit, "X_train_")) or refit:
            self.gpm_fit.fit(self.y_true_train.x, self.y_true_train.y)

    @Plotter
    def plot_prior(self, add_offset=False, title="Samples from Prior Distribution", ax=None):
        data_prior = self.y_true_samples

        if add_offset:
            data_prior = [data + self.offset for data in data_prior]

        plot_lim = 30

        if isinstance(self.y_true_samples, list):
            y = np.vstack([data.y for data in data_prior])

        data0 = data_prior[0]

        y_prior_std = data0.y_std
        ylim = None
        if max(y_prior_std) > plot_lim:
            ylim = [np.min(y[0, :]) - plot_lim, np.max(y[0, :]) + plot_lim]
        plot_gpr_samples(data0.x, data0.y_mean, y_prior_std, y=y, ylim=ylim, ax=ax)
        ax.set_title(title)

    @Plotter
    def plot_posterior(self, add_offset=False, title="Predictive Distribution", ax=None):
        data_dict = {"f_post": self.f_post, "y_true_subsampled": self.y_true_train,
                     "f_true": self.f_true_post}
        if add_offset:
            data_dict = {k: v + self.offset for k, v in data_dict}

        plot_posterior(self.x, data_dict["f_post"].y_mean, y_post_std=data_dict["f_post"].y_std,
                       x_red=data_dict["y_true_subsampled"].x, y_red=data_dict["y_true_subsampled"].y,
                       y_true=data_dict["f_true"].y, ax=ax)
        ax.set_title(title)

    @Plotter
    def plot_true_with_samples(self, add_offset=True, ax=None):
        data_true = self.y_true
        data = self.y_true_train
        if add_offset:
            data_true += self.offset
            data += self.offset

        ax.plot(self.x, data_true.y_mean, "r:")
        ax.scatter(data.x, data.y, color="red", zorder=5, label="Observations")

    @Plotter
    def plot_overall_mean(self, ax=None):
        self.plot_true_with_samples(ax=ax, add_offset=True)
        plot_lines_dict = {"sample mean":  self.data_mean.y_mean, "true mean": self.y_true.y_mean,
                           "predicted_mean": self.f_post.y_mean}
        # loosely dashed, dashed dotted, dotted
        linestyles = [(0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 10))]
        for i, (k, v) in enumerate(plot_lines_dict.items()):
            ax.plot(self.x, np.repeat(np.mean(v) + self.offset, len(self.x)), label=k, linestyle=linestyles[i])
        ax.legend()

    def plot(self, add_offset=False):
        nrows = 3
        ncols = 2

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 6))

        self.plot_prior(add_offset=add_offset, ax=ax[0, 0])

        plot_kernel_function(self.x, self.kernel_sim, ax=ax[0, 1])

        self.fit()

        self.plot_posterior(add_offset=add_offset, ax=ax[1, 0])

        plot_kernel_function(self.x, self.gpm_fit.kernel_, ax=ax[1, 1])

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            ax[2, 0].plot(self.x, v, label=k)
        ax[2, 0].legend()

        for k, v in self.gpm_fit.predict_mean_decomposed(self.x).items():
            ax[2, 1].plot(self.x, v, label=k)

        eval_dict = self.evaluate()

        fig.tight_layout()
        figfile = f"fit_{self.figname_suffix}"
        fig.savefig(self.output_path / f"{figfile}.pdf")
        pd.DataFrame([eval_dict]).to_csv(self.output_path / f"{figfile}.csv")
        plt.close()

    def plot_errors(self):
        nrows = 2
        ncols = 2
        current_row = 0
        current_col = 0
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*10, ncols*10))
        self.plot_posterior(ax=axs[current_row, current_col], add_offset=False)

        for k, v in self.eval_config.items():
            if current_col == (ncols-1):
                current_row += 1
                current_col = 0
            else:
                current_col += 1
            ax = axs[current_row, current_col]
            eval_kwargs = {data_name: (data if v["idx"] is None else data[v["idx"]]) for data_name, data in
                           self.eval_data.items()}
            GPEvaluator(**eval_kwargs).plot_errors(ax=ax)
            ax.set_title(k)

        fig.tight_layout()
        figfile = f"err_{self.figname_suffix}"
        fig.savefig(self.output_path / f"{figfile}.pdf")
        plt.close()

    @property
    def eval_config(self):
        eval_config = {"train_perf": {"idx": self.train_idx, "fun": "evaluate_fun"},
                       "test_perf": {"idx": self.test_idx, "fun": "evaluate_fun"},
                       "overall_perf": {"idx": None, "fun": "evaluate"}}
        return eval_config

    @property
    def eval_data(self):
        return {"y_true": self.y_true.y, "f_true": self.f_true_post, "f_pred": self.f_post}

    def evaluate(self):
        eval_base_kwargs = {"meas_noise_var": self.meas_noise_var}

        param_error = {}
        if self.kernel_sim == self.kernel_fit:
            param_error = {k: (v-self.param_sim[k])/abs(self.param_sim[k]) for k, v in self.param_fit.items()}

        output_dict = {"param_error": param_error}
        for k, v in self.eval_config.items():
            eval_kwargs = {data_name: (data if v["idx"] is None else data[v["idx"]]) for data_name, data in
                           self.eval_data.items()}
            gpe = GPEvaluator(**eval_kwargs, **eval_base_kwargs)
            output_dict[k] = getattr(gpe, v["fun"])()

        output_dict["train_perf"]["log_marginal_likelihood"] = self.gpm_fit.log_marginal_likelihood()
        output_dict["overall_perf_mean"] = GPEvaluator(self.y_true.y, self.f_true_post,
                                                       self.data_mean, meas_noise_var=0).evaluate()

        return output_dict

    def evaluate_multisample(self, n_samples=100):
        current_init_kwargs = {k: v for k, v in vars(self).items() if k in
                               inspect.signature(self.__init__).parameters.keys()}
        current_init_kwargs["normalize_kernel"] = False
        eval_dict = {}

        for i in range(n_samples):
            gps = self.__class__(**current_init_kwargs)
            gps.fit()
            eval_sample = gps.evaluate()
            eval_dict = {k: [v] + eval_dict.get(k, []) for k, v in eval_sample.items()}

        summary_dict = {k: pd.DataFrame(v).mean(axis=0).to_dict() for k, v in eval_dict.items()}
        for v in summary_dict.values():
            v["data_fraction"] = gps.data_fraction
            v["kernel_sim"] = gps.kernel_sim
            v["kernel_fit"] = gps.gpm_fit.kernel_
            v["n_samples"] = n_samples
            v["meas_noise_var"] = gps.meas_noise_var
            v["session_name"] = gps.session_name
        return summary_dict

    @Plotter
    def mean_decomposition_plot(self, ax=None):
        y_post_mean = self.f_true_post.y_mean
        assert np.all(y_post_mean - self.y_true.y < 0.00000001)
        decomposed_dict_sim = self.gpm_sim.predict_mean_decomposed(self.x)
        ax.plot(self.y_true.x, self.y_true.y)
        sum_of_v = np.zeros(len(self.x))
        for k, v in decomposed_dict_sim.items():
            ax.plot(self.x, v, label=k, linestyle="dashed")
            sum_of_v += v

        assert np.all(sum_of_v - y_post_mean < 0.00000001)
        # ax.plot(data.x, sum_of_v, label="sum")
        # ax.plot(data.x, y_post_mean, label="post_mean")
        ax.legend()


def plot_mean_decompose(kernel="sin_rbf"):
    gpm = GPSimulator(kernel_sim=OU_KERNELS["fixed"][kernel], **base_config)
    gpm.mean_decomposition_plot(figname=kernel)


if __name__ == "__main__":
    setup_logging()

