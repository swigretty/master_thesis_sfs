import datetime
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
                 meas_noise=0, kernel_fit=None, normalize_y=False, output_path=OUTPUT_PATH,
                 data_fraction_weights=None, data_fraction=0.3, data_true=None, rng=None, normalize_kernel=False,
                 session_name=None):

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.x = x
        self.mean_f = mean_f
        self.meas_noise = meas_noise
        self.offset = np.mean([mean_f(xi) for xi in self.x])

        self._data_fraction_weights = data_fraction_weights
        self.data_fraction = data_fraction

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.kernel_sim = kernel_sim
        if normalize_kernel:
            self.kernel_sim, self.meas_noise = self.get_normalized_kernel(kernel_sim,
                                                                          meas_noise=self.meas_noise)

        if kernel_fit is None:
            kernel_fit = self.kernel_sim

        self.kernel_fit = kernel_fit

        # self.kernel_sim_meas_noise = self.kernel_sim + WhiteKernel(noise_level=self.meas_noise,
        #                                                            noise_level_bounds="fixed")
        #
        # self.gpm_sim_meas_noise = GPR(kernel=self.kernel_sim_meas_noise, normalize_y=False, optimizer=None,
        #                               rng=rng, alpha=0)
        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None, rng=rng, alpha=self.meas_noise)
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=normalize_y, alpha=self.meas_noise, rng=rng)

        self.data_true = data_true

        self.session_name = session_name
        if self.session_name is None:
            self.session_name = self.kernel_sim.__class__.__name__
        self.figname_suffix = f"{self.session_name}_{self.data_fraction:.2f}"

        logger.info(f"Initialized {self.__class__.__name__} with \n {kernel_sim=} \n {kernel_fit=}")

    def sim_gp(self, n_samples=5):
        # samples without measurement noise
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
            self.x, n_samples=n_samples, predict_y=True)  # mean_f=self.mean_f

        # if self.meas_noise:
        #     # TODO should this be scaled as well ?
        #     # y_prior = y_prior + self.meas_noise * np.std(y_prior, axis=0) * np.random.standard_normal((y_prior.shape))
        #     y_prior = y_prior + self.meas_noise * self.rng.standard_normal((y_prior.shape))
        #     y_prior_cov[np.diag_indices_from(y_prior_cov)] += self.meas_noise

        data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov) for idx in
                range(n_samples)]

        return data

    @property
    def data_fraction_weights(self):
        if self._data_fraction_weights != "seasonal":
            return self._data_fraction_weights

        self.gpm_sim.fit(self.data_true.x, self.data_true.y)

        true_dec = self.gpm_sim.predict_mean_decomposed(self.x)
        fun = [fun for name, fun in true_dec.items() if "ExpSineSquared" in name]
        assert len(fun) == 1, "cannot extract seasonal pattern"
        weights = fun[0] - min(fun[0])
        return weights * 0.1

    @property
    def data_true_post_y(self):
        y_post_cov = self.data_true_post + np.diag(np.repeat(self.meas_noise, len(self.data_true_post)))
        return GPData(x=self.data_true_post.x, y=self.data_true_post.y, y_mean=self.data_true_post.y_mean,
                      y_cov=y_post_cov)

    @cached_property
    def data_true_post(self):
        self.gpm_sim.fit(self.data_true.x, self.data_true.y)
        y_post_mean, y_post_cov = self.gpm_sim.predict(self.data_true.x, return_cov=True)
        # y_post_mean will only diverge from self.data_true.y if meas_noise != 0
        if not self.meas_noise:
            assert np.mean((self.data_true.y - y_post_mean)**2) < 10**(-4)
        return GPData(x=self.data_true.x, y=self.data_true.y, y_mean=y_post_mean, y_cov=y_post_cov)

    @property
    def data_true(self):
        return self._data_true

    @data_true.setter
    def data_true(self, data=None):
        self._data_true = data
        if data is None:
            self._data_true = self.sim_gp(n_samples=1)[0]

    @cached_property
    def data(self):
        return self.subsample_data(self.data_true, self.data_fraction, data_fraction_weights=self.data_fraction_weights)

    @property
    def data_prior(self):
        if hasattr(self, "_data_prior"):
            return self._data_prior
        self.data_prior = self.sim_gp(n_samples=5)
        return self._data_prior

    @data_prior.setter
    def data_prior(self, data: list):
        data = data + [self.data_true]
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
    def data_post(self):
        y_post, y_post_mean, y_post_cov = self.gpm_fit.sample_from_posterior(self.x, n_samples=1, predict_y=False)
        return GPData(x=self.x, y=y_post, y_mean=y_post_mean, y_cov=y_post_cov)

    @cached_property
    def data_post_y(self):
        """
        @cached_property
        def data_post_y(self):
            y_cov = copy(self.data_post.y_cov)
            y_cov[np.diag_indices_from(y_cov)] += self.meas_noise
            return GPData(x=self.x, y=self.data_post.y, y_mean=self.data_post.y_mean, y_cov=y_cov)

        """
        y_post_cov = self.data_post.y_cov + np.diag(np.repeat(self.meas_noise, len(self.data_post)))
        return GPData(x=self.data_post.x, y=self.data_post.y, y_mean=self.data_post.y_mean,
                      y_cov=y_post_cov)

    @property
    def data_mean(self):
        """
        Using the overall mean as prediction.
        cov is calculated assuming iid data.
        """
        sigma_mean = 1/len(self.data.y) * np.var(self.data.y)
        y_cov = np.zeros((len(self.x), len(self.x)), float)
        np.fill_diagonal(y_cov, sigma_mean)
        y = np.repeat(np.mean(self.data.y), len(self.x))
        return GPData(x=self.x, y=y, y_mean=y, y_cov=y_cov)

    @property
    def test_idx(self):
        """
        or
        x = np.setdiff1d(self.data_post.x, self.data.x, assume_unique=True)
        return np.arange(len(self.x))[np.isin(self.x, x)]

        """
        return [idx for idx in range(len(self.x)) if self.x[idx] not in
                self.data.x]

    @property
    def train_idx(self):
        return [idx for idx in range(len(self.x)) if self.x[idx] in
                self.data.x]

    @staticmethod
    def subsample_data(data: GPData, data_fraction: float, data_fraction_weights=None):
        if callable(data_fraction_weights):
            weights = data_fraction_weights(data.y)
        else:
            weights = data_fraction_weights
        idx = get_red_idx(len(data), data_fraction=data_fraction, weights=weights)
        return data[idx]

    @staticmethod
    def extract_params_from_kernel(kernel):
        return {k: v for k, v in kernel.get_params().items() if ("__" in k) and ("bounds" not in k) and any(
            [pn in k for pn in PARAM_NAMES])}

    @classmethod
    def get_normalized_kernel(cls, kernel, meas_noise=0):
        previousloglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)

        kernel_ = copy(kernel)
        std_range = (0.95, 1.05)

        i = 0
        scale = 1
        y_std = 2

        while (std_range[0] > y_std) or (
                std_range[1] < y_std):
            gps = cls(kernel_sim=kernel_, meas_noise=meas_noise*1/scale)
            data_sim = gps.sim_gp(n_samples=100)
            y_var = np.mean([np.var(d.y) for d in data_sim])
            # logger.info(y_std)
            scale *= y_var
            kernel_ = ConstantKernel(constant_value=1/scale, constant_value_bounds="fixed") * kernel
            i += 1
            if i > 20:
                break

        logger.setLevel(previousloglevel)

        logger.info(f"final kernel {kernel_} with scaling {1/scale} and {y_std=}")

        return kernel_, 1/scale * meas_noise

    def choose_sample_from_prior(self, data_index: int = 0):
        data_single = copy(self.data_prior)
        data_single.y = self.data_prior.y[:, data_index]
        return data_single

    def fit(self, refit=False):
        if (not hasattr(self.gpm_fit, "X_train_")) or refit:
            self.gpm_fit.fit(self.data.x, self.data.y)

    @Plotter
    def plot_prior(self, add_offset=False, title="Samples from Prior Distribution", ax=None):
        data_prior = self.data_prior

        if add_offset:
            data_prior = [data + self.offset for data in data_prior]

        plot_lim = 30

        if isinstance(self.data_prior, list):
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
        data_post = self.data_post
        data = self.data
        data_true = self.data_true_post

        if add_offset:
            data_post += self.offset
            data += self.offset
            data_true += self.offset

        plot_posterior(data_post.x, data_post.y_mean, y_post_std=data_post.y_std, x_red=data.x, y_red=data.y,
                       y_true=data_true.y_mean, ax=ax)
        ax.set_title(title)

    @Plotter
    def plot_true_with_samples(self, add_offset=True, ax=None):
        data_true = self.data_true_post
        data = self.data
        if add_offset:
            data_true += self.offset
            data += self.offset

        ax.plot(self.x, data_true.y_mean, "r:")
        ax.scatter(data.x, data.y, color="red", zorder=5, label="Observations")

    @Plotter
    def plot_overall_mean(self, ax=None):
        self.plot_true_with_samples(ax=ax, add_offset=True)
        plot_lines_dict = {"sample mean":  self.data_mean.y_mean, "true mean": self.data_true.y,
                           "predicted_mean": self.data_post.y_mean}
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
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*10, ncols*10))
        self.plot_posterior(ax=axs[0, 0], add_offset=False)

        error_plot_dict = {"errors overall": {"idx": None, "ax": axs[0, 1]},
                           "errors_train": {"idx": self.train_idx, "ax": axs[1, 0]},
                           "errors_test": {"idx": self.test_idx, "ax": axs[1, 1]}}

        for k, v in error_plot_dict.items():
            data_true = self.data_true_post
            data_post = self.data_post
            if v["idx"] is not None:
                data_true = data_true[v["idx"]]
                data_post = data_post[v["idx"]]
            GPEvaluator(data_true, data_post).plot_errors(ax=v["ax"])
            v["ax"].set_title(k)

        fig.tight_layout()
        figfile = f"err_{self.figname_suffix}"
        fig.savefig(self.output_path / f"{figfile}.pdf")
        plt.close()

    def evaluate(self, predict_y=False):
        param_error = {}
        data_post = self.data_post
        if predict_y:
            data_post = self.data_post_y

        if self.kernel_sim == self.kernel_fit:
            param_error = {k: (v-self.param_sim[k])/abs(self.param_sim[k]) for k, v in self.param_fit.items()}

        train_perf = GPEvaluator(self.data_true_post[self.train_idx], data_post[self.train_idx]).evaluate_fun()
        train_perf["log_marginal_likelihood"] = self.gpm_fit.log_marginal_likelihood()
        test_perf = GPEvaluator(self.data_true_post[self.test_idx], data_post[self.test_idx]).evaluate_fun()

        overall_perf = GPEvaluator(self.data_true_post, data_post).evaluate()

        overall_perf_mean = GPEvaluator(self.data_true_post, self.data_mean).evaluate()

        return {"param_error": param_error, "train_perf": train_perf, "test_perf": test_perf,
                "overall_perf": overall_perf, "overall_perf_mean": overall_perf_mean}

    def evaluate_multisample(self, n_samples=100):
        gps = copy(self)
        samples = gps.sim_gp(n_samples)
        eval_dict = {}

        for sample in samples:
            gps.data_true = sample
            try:
                gps.fit(refit=True)
            except Exception as e:
                logger.warning("Could not fit GP")
            eval_sample = gps.evaluate()
            eval_dict = {k: [v] + eval_dict.get(k, []) for k, v in eval_sample.items()}

        summary_dict = {k: pd.DataFrame(v).mean(axis=0).to_dict() for k, v in eval_dict.items()}
        for v in summary_dict.values():
            v["data_fraction"] = gps.data_fraction
            v["kernel_sim"] = gps.kernel_sim
            v["kernel_fit"] = gps.gpm_fit.kernel_
            v["n_samples"] = n_samples
        return summary_dict

    @Plotter
    def mean_decomposition_plot(self, ax=None):
        y_post_mean = self.data_true_post.y_mean
        assert np.all(y_post_mean - self.data_true.y < 0.00000001)
        decomposed_dict_sim = self.gpm_sim.predict_mean_decomposed(self.x)
        ax.plot(self.data_true.x, self.data_true.y)
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

