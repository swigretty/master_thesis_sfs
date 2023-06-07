import datetime
from dataclasses import dataclass, asdict
from functools import lru_cache
from copy import copy
import pandas as pd
from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from itertools import permutations
from exploration.gp import GPR, plot_gpr_samples, plot_kernel_function, plot_posterior
from exploration.constants import OUTPUT_PATH
from exploration.explore import get_red_idx, get_sesonal_weights
from exploration.simulate_gp_config import base_config, OU_KERNELS, KERNELS, PARAM_NAMES, PERIOD_DAY
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')

permutations(iterable, r=None)¶

@dataclass
class GPData():
    """
    gp1 = GPData(x=np.array([1]), y = np.array([2]))
    """
    x: np.array
    y: np.array
    y_mean: np.array = None
    y_cov: np.array = None

    def __len__(self):
        return len(self.x)

    def to_df(self):
        return pd.DataFrame(asdict(self))

    @classmethod
    def from_df(cls, df):
        return cls(**df.to_dict())

    def __getitem__(self, idx):
        new_data = self.__class__(
            **{k: (np.array([[v[io, ii] for ii in idx] for io in idx]) if v.ndim == 2 else v[idx])
               for k, v in asdict(self).items() if v is not None})
        return new_data

def weights_func_value(y):
    return y - np.min(y)


class GPSimulator():

    def __init__(self, x=np.linspace(0, 40, 200), kernel_sim=1 * Matern(nu=0.5, length_scale=1), mean_f=lambda x: 120,
                 meas_noise=0, kernel_fit=None, normalize_y=False, output_path=OUTPUT_PATH,
                 data_fraction_weights=None, data_fraction=0.3, data_true=None):

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.output_path = output_path

        self.x = x
        self.kernel_sim = kernel_sim

        self.mean_f = mean_f
        self.meas_noise = meas_noise
        self.offset = mean_f(0)

        self.data_fraction_weights = data_fraction_weights
        self.data_fraction = data_fraction
        self.data_true = data_true

        if kernel_fit is None:
            kernel_fit = kernel_sim
            # TODO only allow for centered input
            # if not normalize_y:
            #     kernel_fit = kernel_sim + ConstantKernel(constant_value=self.offset ** 2, constant_value_bounds="fixed")
        self.kernel_fit = kernel_fit

        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None)
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=normalize_y, alpha=self.meas_noise)

        logger.info(f"Initialized {self.__class__.__name__} with \n {kernel_sim=} \n {kernel_fit=}")

    def sim_gp(self, n_samples=5):
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
            self.x, n_samples=n_samples, mean_f=self.mean_f)

        # if n_samples == 1:
        #     y_prior = y_prior.reshape(-1)
        #     y_prior_mean = y_prior_mean.reshape(-1)

        if self.meas_noise:
            # TODO should this be scaled as well ?
            # y_prior = y_prior + self.meas_noise * np.std(y_prior, axis=0) * np.random.standard_normal((y_prior.shape))
            y_prior = y_prior + self.meas_noise * np.random.standard_normal((y_prior.shape))
            y_prior_cov[np.diag_indices_from(y_prior_cov)] += self.meas_noise

        data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean[:, idx], y_cov=y_prior_cov) for idx in
                range(n_samples)]

        return data

    @property
    def data_true(self):
        return self._data_true

    @data_true.setter
    def data_true(self, data=None):
        self._data_true = data
        if data is None:
            self._data_true = self.sim_gp(n_samples=1)[0]

    @property
    @lru_cache()
    def data(self):
        return self.subsample_data(self.data_true, self.data_fraction, data_fraction_weights=self.data_fraction_weights)

    @property
    def data_prior(self):
        if hasattr(self, "_data_prior"):
            return self._data_prior
        self.data_prior = self.sim_gp(n_samples=5)
        return self._data_prior

    @data_prior.setter
    def data_prior(self, data):
        data.y = np.hstack(self.data_true.y, data.y)
        self._data_prior = data

    @property
    def pram_sim(self):
        return self.extract_params_from_kernel(self.kernel_sim)

    @property
    def param_fit(self):
        if not hasattr(self.gpm_fit, "X_train_"):
            raise "GP has not been fitted yet"
        return self.extract_params_from_kernel(self.gpm_fit.kernel_)

    @property
    @lru_cache()
    def data_post(self):
        y_post, y_post_mean, y_post_cov = self.gpm_fit.sample_from_posterior(self.x, n_samples=1)
        return GPData(x=self.x, y=y_post, y_mean=y_post_mean, y_cov=y_post_cov)

    @property
    def test_idx(self):
        x = np.setdiff1d(self.data_post.x, self.data, assume_unique=True)
        return self.x == x

    @property
    def train_idx(self):
        return self.x == self.data.x

    @staticmethod
    def subsample_data(data: GPData, data_fraction: float, data_fraction_weights=None):
        if callable(data_fraction_weights):
            weights = data_fraction_weights(data.y)
        else:
            weights = data_fraction_weights
        idx = get_red_idx(len(data), data_fraction=data_fraction, weights=weights)
        return data[idx]

    @staticmethod
    def calculate_ci(se, mean, alpha=0.05, dist=norm):
        return (mean - se * dist.ppf(1 - alpha/2), mean + se * dist.ppf(1 - alpha/2))

    @staticmethod
    def se_avg(y_post_cov):
        return 1 / y_post_cov.shape[0] * np.sqrt(np.sum(y_post_cov))

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
            y_std = np.std(data_sim.y)
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

    def fit(self):
        self.gpm_fit.fit(self.data.x, self.data.y)

    def plot_prior(self, ax):
        plot_lim = 30

        if isinstance(self.data_prior, list):
            y = np.vstack(data.y for data in self.data_prior)

        data0 = self.data_prior[0]

        y_prior_std = np.diag(data0.y_cov)
        ylim = None
        if max(y_prior_std) > plot_lim:
            ylim = [np.min(y[0, :]) - plot_lim, np.max(y[0, :]) + plot_lim]
        plot_gpr_samples(ax, data0.x, data0.y_mean, y_prior_std, y=y, ylim=ylim)
        ax.set_title("Samples from Prior Distribution")

    def plot_posterior(self, ax):
        plot_posterior(ax, self.data_post.x, self.data_post.y_mean, np.diag(self.data_post.y_cov), self.data.x,
                       self.data.y, y_true=self.data_true.y)
        ax.set_title("Predictive Distribution")

    def plot(self, figname=None):
        nrows = 3
        ncols = 2

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))

        self.plot_prior(ax[0,0])
        plot_kernel_function(ax[0, 1], self.x, self.kernel_sim)

        self.plot_posterior(ax[1, 0])
        plot_kernel_function(ax[1, 1], self.x, self.gpm_fit.kernel_)

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            ax[2, 0].plot(self.x, v, label=k)
        ax[2, 0].legend()

        for k, v in self.gpm_fit.predict_mean_decomposed(self.x).items():
            ax[2, 1].plot(self.x, v, label=k)

        fig.tight_layout()

        if figname is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_path / f"{figname}_{self.data_fraction:.2f}.pdf")

    def evaluate_ci_for_overall_mean(self):
        covered = False
        ci = self.calculate_ci(self.se_avg(self.data_post.y_cov), np.mean(self.data_post.y_mean))
        if ci[0] < np.mean(self.data_true.y) < ci[1]:
            covered = True
        return {"ci": ci, "covered": covered, "ci_width": ci[1]-ci[0]}

    def evaluate(self):
        ci_overall_mean = self.evaluate_ci_for_overall_mean()
        param_error = {k: (v-self.param_sim[k])/abs(self.param_sim[k]) for k, v in self.param_fit.items()}

        pred_prob_test = self.get_predictive_probability_fit(self.data_post[self.test_idx],
                                                             self.data_true.y[self.test_idx])
        return

    def get_predictive_probability_fit(self, data_predict: GPData, y):
        return multivariate_normal.pdf(y, mean=data_predict.y_mean, cov=data_predict.y_cov)

    def sim_and_assess_performance(self, n_samples=100, data_fraction=0.3):
        """
        simulate from the prior,
        then simulate from the model using those values from the prior, and
        estimate the parameters using the same prior.
        """
        data_prior = self.sim_gp(n_samples=n_samples)
        n_ci_coverage = 0
        ci_array = np.zeros((n_samples, 2))

        param_fit_list = []
        param_sim = self.extract_params_from_kernel(self.kernel_sim)
        true_mean_list = []

        for sample_index in range(n_samples):
            data_true = self.choose_sample_from_prior(data_prior, data_index=sample_index)
            data = self.subsample_data_sim(data_true, data_fraction=data_fraction)
            self.fit(data)

            y_post_mean, y_post_cov = self.gpm_fit.predict(self.x, return_cov=True)
            ci = self.calculate_ci(self.se_avg(y_post_cov), np.mean(y_post_mean))
            ci_array[sample_index, :] = ci

            kernel_fit = self.gpm_fit.kernel_
            param_fit_list.append(self.extract_params_from_kernel(kernel_fit))

            true_mean = np.mean(data_true.y)
            true_mean_list.append(true_mean)

            if ci[0] < true_mean < ci[1]:
                n_ci_coverage += 1
            else:
                logger.info(f"true mean {np.mean(data_true.y)} not covered by confidence intervals {ci} \n "
                            f"{kernel_fit=} vs. {self.kernel_sim=}")

        param_fit_df = pd.DataFrame(param_fit_list)
        param_fit_mean = param_fit_df.mean(axis=0).to_dict()
        param_rel_error = {k: (v-param_sim[k])/abs(param_sim[k]) for k, v in param_fit_mean.items()}
        param_rel_error = {k: v for k, v in param_rel_error.items() if v > 0.000001}
        ci_coverage = n_ci_coverage/n_samples
        mean_ci_width = np.mean(np.diff(ci_array, axis=1))
        return {"data_fraction": data_fraction, "ci_coverage": ci_coverage,
                "mean_ci_width": mean_ci_width, "kernel_sim": self.kernel_sim, "param_rel_error": param_rel_error,
                "true_mean_mean": np.mean(true_mean_list), "true_mean_std": np.std(true_mean_list)}

    def mean_decomposition_plot(self, figname=None):
        data = self.sim_gp(n_samples=1)
        self.gpm_sim.fit(data.x, data.y)

        y_post, y_post_mean, y_post_cov = self.gpm_sim.sample_from_posterior(self.x, n_samples=1)
        decomposed_dict_sim = self.gpm_sim.predict_mean_decomposed(self.x)

        nrows = 1
        ncols = 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))

        ax.plot(data.x, data.y)

        sum_of_v = np.zeros(len(self.x))
        for k, v in decomposed_dict_sim.items():
            ax.plot(self.x, v, label=k, linestyle="dashed")
            sum_of_v += v

        assert np.all(sum_of_v - y_post_mean < 0.00000001)
        assert np.all(y_post_mean - data.y < 0.00000001)
        # ax.plot(data.x, sum_of_v, label="sum")
        # ax.plot(data.x, y_post_mean, label="post_mean")
        ax.legend()

        fig.tight_layout()

        if figname is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_path / f"{figname}_mean_dec.pdf")


def plot_mean_decompose(kernel="sin_rbf"):
    gpm = GPSimulator(kernel_sim=OU_KERNELS["fixed"][kernel], **base_config)
    gpm.mean_decomposition_plot(figname=kernel)


if __name__ == "__main__":
    setup_logging()

    plot_mean_decompose()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    kernels_limited = list(OU_KERNELS["bounded"].keys())
    kernels_limited = [k for k in kernels_limited if k in ["sinrbf_rbf"]]
    data_fraction_weights = get_sesonal_weights(base_config["x"], period=PERIOD_DAY)

    modes = {
        # "ou_fixed": {"kernels": OU_KERNELS["fixed"],
        #                   "config": {"normalize_y": False, **base_config}},
             "ou_bounded": {"kernels":  OU_KERNELS["bounded"],
                            "config": {"normalize_y": False, **base_config}},
        #     "ou_bounded_normk_non_uniform": {"kernels": OU_KERNELS["bounded"],
        #                       "config": {"normalize_y": False, "data_fraction_weights": weights_func_value,
        #                                  **base_config}},
        #     "ou_bounded_normk_cycl": {"kernels": OU_KERNELS["bounded"], "config": {
        #         "normalize_y": False, "data_fraction_weights": data_fraction_weights,
        #         **base_config}}
        #      "ou_bounded_nonorm": {"kernels": OU_KERNELS["bounded"],
        #                    "config": {"normalize_y": False, **base_config}},
        # #      "ou_unbounded":
        # #          {"kernels": OU_KERNELS["unbounded"],
        # #           "config": {"normalize_y": True, **base_config}},
        #
        #      "bounded_nonorm": {"kernels": KERNELS["bounded"],
        #                         "config": {"normalize_y": False, **base_config}},
             # "bounded": {"kernels": KERNELS["bounded"],
             #             "config": {"normalize_y": True, **base_config}}
             }

    for mode_name, mode_config in modes.items():
        ci_info = []
        for k_name, k in mode_config["kernels"].items():
            if k_name not in kernels_limited:
                continue
            start = datetime.datetime.utcnow()
            logger.info(f"Simulation started for {mode_name}: {k_name}")
            k_norm = GPSimulator.get_normalized_kernel(k)
            gps = GPSimulator(kernel_sim=k_norm, **mode_config["config"])

            gps.sim_fit_plot(figname=f"gp_{k_name}_{mode_name}")
        #     output_dict = gps.test_ci(data_fraction=0.3)
        #     ci_info.append({**output_dict, **mode_config["config"]})
        #
        #     logger.info(f"Simulation ended for {mode_name}: {k_name}. "
        #                 f"Duration: {(datetime.datetime.utcnow()-start).total_seconds()} sec")
        #
        # ci_info_df = pd.DataFrame(ci_info)
        # ci_info_df = ci_info_df[[col for col in ci_info_df.columns if col not in ["x", 'mean_f']]]
        # ci_info_df.to_csv(OUTPUT_PATH / f"ci_info_{mode_name}.csv")


