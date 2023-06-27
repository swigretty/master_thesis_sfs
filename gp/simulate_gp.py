import datetime
from functools import cached_property
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
import scipy
import matplotlib as mpl
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from gp.gp_regressor import GPR
from gp.gp_plotting_utils import plot_kernel_function, plot_posterior, plot_gpr_samples, ts_plotter
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
        self.kernel_sim = kernel_sim
        if normalize_kernel:
            self.kernel_sim = self.get_normalized_kernel(kernel_sim)
        self.session_name = session_name
        if self.session_name is None:
            self.session_name = self.kernel_sim.__name__


        self.mean_f = mean_f
        self.meas_noise = meas_noise
        self.offset = np.mean([mean_f(xi) for xi in self.x])

        self._data_fraction_weights = data_fraction_weights
        self.data_fraction = data_fraction

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        if kernel_fit is None:
            kernel_fit = self.kernel_sim
            # TODO only allow for centered input
            # if not normalize_y:
            #     kernel_fit = kernel_sim + ConstantKernel(constant_value=self.offset ** 2,
            #     constant_value_bounds="fixed")
        self.kernel_fit = kernel_fit

        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None, rng=rng, alpha=0)
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=normalize_y, alpha=self.meas_noise, rng=rng)

        self.data_true = data_true

        logger.info(f"Initialized {self.__class__.__name__} with \n {kernel_sim=} \n {kernel_fit=}")

    def plotter(self):
        return ts_plotter(output_path=self.output_path)

    def sim_gp(self, n_samples=5):
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
            self.x, n_samples=n_samples)  # mean_f=self.mean_f

        # if n_samples == 1:
        #     y_prior = y_prior.reshape(-1)
        #     y_prior_mean = y_prior_mean.reshape(-1)

        if self.meas_noise:
            # TODO should this be scaled as well ?
            # y_prior = y_prior + self.meas_noise * np.std(y_prior, axis=0) * np.random.standard_normal((y_prior.shape))
            y_prior = y_prior + self.meas_noise * self.rng.standard_normal((y_prior.shape))
            y_prior_cov[np.diag_indices_from(y_prior_cov)] += self.meas_noise

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
        y_post, y_post_mean, y_post_cov = self.gpm_fit.sample_from_posterior(self.x, n_samples=1)
        return GPData(x=self.x, y=y_post, y_mean=y_post_mean, y_cov=y_post_cov)

    @property
    def data_mean(self):
        sigma_mean = 1/len(self.data.y) * np.var(self.data.y)
        y_cov = np.zeros((len(self.x), len(self.x)), float)
        np.fill_diagonal(y_cov, sigma_mean)
        y = np.repeat(np.mean(self.data.y), len(self.x))
        return GPData(x=self.x, y=y, y_mean=y, y_cov=y_cov)

    @cached_property
    def data_post_y(self):
        y_cov = copy(self.data_post.y_cov)
        y_cov[np.diag_indices_from(y_cov)] += self.meas_noise
        return GPData(x=self.x, y=self.data_post.y, y_mean=self.data_post.y_mean, y_cov=y_cov)

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
    def get_normalized_kernel(cls, kernel):
        kernel_ = copy(kernel)
        std_range = (0.95, 1.05)
        i = 0
        scale = 1
        y_std = 2
        while (std_range[0] > y_std) or (std_range[1] < y_std):
            gps = cls(kernel_fit=kernel_)
            data_sim = gps.sim_gp(n_samples=100)
            y_std = np.mean([np.std(d.y) for d in data_sim])
            # logger.info(y_std)
            scale *= y_std
            kernel_ = ConstantKernel(constant_value=1/scale, constant_value_bounds="fixed") * kernel
            i += 1
            if i > 20:
                break
        logger.info(f"final kernel {kernel_} with scaling {1/scale} and {y_std=}")
        return kernel_

    def choose_sample_from_prior(self, data_index: int = 0):
        data_single = copy(self.data_prior)
        data_single.y = self.data_prior.y[:, data_index]
        return data_single

    def fit(self, refit=False):
        if (not hasattr(self.gpm_fit, "X_train_")) or refit:
            self.gpm_fit.fit(self.data.x, self.data.y)

    @plotter()
    def plot_prior(self, add_offset=False, title="Samples from Prior Distribution", ax=None, figname_suffix=""):
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
        ax[0, 0].set_title(title)

    @plotter()
    def plot_posterior(self, add_offset=False, title="Predictive Distribution", ax=None, **kwargs):
        data_post = self.data_post
        data = self.data
        data_true = self.data_true

        if add_offset:
            data_post += self.offset
            data += self.offset
            data_true += self.offset

        plot_posterior(data_post.x, data_post.y_mean, y_post_std=data_post.y_std, x_red=data.x, y_red=data.y,
                       y_true=data_true.y, ax=ax)
        ax.set_title(title)

    @plotter()
    def plot_true_with_samples(self, add_offset=True, ax=None, **kwargs):
        data_true = self.data_true
        data = self.data
        if add_offset:
            data_true += self.offset
            data += self.offset

        ax.plot(self.x, data_true.y, "r:")
        ax.scatter(data.x, data.y, color="red", zorder=5, label="Observations")

        figname_suffix = kwargs.get("figname_suffix", "")
        kwargs["figname_suffix"] = f"{figname_suffix}_{self.data_fraction:.2f}"

    def plot_overall_mean(self, ax):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        self.plot_true_with_samples(ax, add_offset=True)
        ax.plot(self.data_mean.x, self.data_mean.y_mean, label="sample mean")
        ax.plot(self.data_true.x, np.mean(self.data_true.y), label="true_mean")
        ax.plot(self.data_post.x, np.mean(self.data_post.y_mean), label="predicted_mean")

    def plot(self, figname=None, add_offset=False):
        nrows = 3
        ncols = 2

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 6))

        self.plot_prior(ax[0, 0], add_offset=add_offset)

        plot_kernel_function(self.x, self.kernel_sim, ax=ax[0, 1])

        self.fit()
        self.gpm_sim.fit(self.data_true.x, self.data_true.y)

        self.plot_posterior(ax[1, 0], add_offset=add_offset)

        plot_kernel_function(self.x, self.gpm_fit.kernel_, ax=ax[1, 1])

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            ax[2, 0].plot(self.x, v, label=k)
        ax[2, 0].legend()

        for k, v in self.gpm_fit.predict_mean_decomposed(self.x).items():
            ax[2, 1].plot(self.x, v, label=k)

        fig.tight_layout()
        eval_dict = self.evaluate()

        if figname is None:
            figname = self.kernel_sim.__name__

        figfile = f"fit_{figname}_{self.data_fraction:.2f}"
        fig.savefig(self.output_path / f"{figfile}.pdf")
        pd.DataFrame([eval_dict]).to_csv(self.output_path / f"{figfile}.csv")
        plt.close()

    def plot_errors(self, figname=None):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
        self.plot_posterior(ax=ax[0, 0], add_offset=False)
        GPEvaluator(self.data_true, self.data_post).plot_errors(ax=ax[0, 1])
        ax[0, 1].set_title("errors overall")
        GPEvaluator(self.data_true[self.train_idx], self.data_post[self.train_idx]).plot_errors(ax=ax[1, 0])
        ax[1, 0].set_title("errors train")
        GPEvaluator(self.data_true[self.test_idx], self.data_post[self.test_idx]).plot_errors(ax=ax[1, 1])
        ax[1, 1].set_title("errors test")

        if figname is not None:
            figfile = f"err_{figname}_{self.data_fraction:.2f}"
            fig.savefig(self.output_path / f"{figfile}.pdf")


    def evaluate(self):
        param_error = {}
        if self.kernel_sim == self.kernel_fit:
            param_error = {k: (v-self.param_sim[k])/abs(self.param_sim[k]) for k, v in self.param_fit.items()}

        train_perf = GPEvaluator(self.data_true[self.train_idx], self.data_post[self.train_idx]).evaluate_fun()
        train_perf["log_marginal_likelihood"] = self.gpm_fit.log_marginal_likelihood()

        test_perf = GPEvaluator(self.data_true[self.test_idx], self.data_post[self.test_idx]).evaluate_fun()

        overall_perf = GPEvaluator(self.data_true, self.data_post).evaluate()

        overall_perf_mean = GPEvaluator(self.data_true, self.data_mean).evaluate()

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

    def mean_decomposition_plot(self, figname=None):

        self.gpm_sim.fit(self.data_true.x, self.data_true.y)

        y_post, y_post_mean, y_post_cov = self.gpm_sim.sample_from_posterior(self.x, n_samples=1)
        decomposed_dict_sim = self.gpm_sim.predict_mean_decomposed(self.x)

        nrows = 1
        ncols = 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))

        ax.plot(self.data_true.x, self.data_true.y)

        sum_of_v = np.zeros(len(self.x))
        for k, v in decomposed_dict_sim.items():
            ax.plot(self.x, v, label=k, linestyle="dashed")
            sum_of_v += v

        assert np.all(sum_of_v - y_post_mean < 0.00000001)
        assert np.all(y_post_mean - self.data_true.y < 0.00000001)
        # ax.plot(data.x, sum_of_v, label="sum")
        # ax.plot(data.x, y_post_mean, label="post_mean")
        ax.legend()

        fig.tight_layout()

        if figname is not None:
            fig.savefig(self.output_path / f"{figname}_mean_dec.pdf")


def plot_mean_decompose(kernel="sin_rbf"):
    gpm = GPSimulator(kernel_sim=OU_KERNELS["fixed"][kernel], **base_config)
    gpm.mean_decomposition_plot(figname=kernel)


if __name__ == "__main__":
    setup_logging()

