import datetime
from dataclasses import dataclass, asdict
from functools import cached_property
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from exploration.gp import GPR, plot_gpr_samples, plot_kernel_function, plot_posterior
from exploration.constants import OUTPUT_PATH
from exploration.explore import get_red_idx, get_sesonal_weights
from exploration.simulate_gp_config import base_config, OU_KERNELS, KERNELS, PARAM_NAMES, PERIOD_DAY
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')


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
        return pd.DataFrame(asdict(self))

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

        if kernel_fit is None:
            kernel_fit = kernel_sim
            # TODO only allow for centered input
            # if not normalize_y:
            #     kernel_fit = kernel_sim + ConstantKernel(constant_value=self.offset ** 2,
            #     constant_value_bounds="fixed")
        self.kernel_fit = kernel_fit

        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None)
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=normalize_y, alpha=self.meas_noise)

        self.data_true = data_true

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

        data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov) for idx in
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
    def calculate_ci(se, mean, alpha=0.05, dist=norm):
        return (mean - se * dist.ppf(1 - alpha/2), mean + se * dist.ppf(1 - alpha/2))

    @staticmethod
    def se_avg(y_post_cov):
        """
        Var(A+B) = Var(A) + Var(B) + 2Cov(A,B)
        Var(c * A) = c^2 * A
        """
        return 1 / y_post_cov.shape[0] * np.sqrt(np.sum(y_post_cov))

    @staticmethod
    def get_predictive_logprob(data_predict: GPData, y: np.typing.ArrayLike):
        # slogdet = np.linalg.slogdet(data_predict.y_cov)
        # prob = np.exp(((-len(y)/2) * np.log(2 * np.pi) - 1/2 * slogdet[1] - 1/2 * (
        #         y - data_predict.y_mean).T @ np.linalg.inv(data_predict.y_cov) @ (y - data_predict.y_mean)))
        prob2 = multivariate_normal.logpdf(y, mean=data_predict.y_mean, cov=data_predict.y_cov)
        return prob2

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

    def plot_prior(self, ax):
        plot_lim = 30

        if isinstance(self.data_prior, list):
            y = np.vstack([data.y for data in self.data_prior])

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

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 6))

        self.plot_prior(ax[0, 0])
        plot_kernel_function(ax[0, 1], self.x, self.kernel_sim)

        self.fit()
        self.gpm_sim.fit(self.data_true.x, self.data_true.y)

        self.plot_posterior(ax[1, 0])
        plot_kernel_function(ax[1, 1], self.x, self.gpm_fit.kernel_)

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            ax[2, 0].plot(self.x, v, label=k)
        ax[2, 0].legend()

        for k, v in self.gpm_fit.predict_mean_decomposed(self.x).items():
            ax[2, 1].plot(self.x, v, label=k)

        fig.tight_layout()
        eval_dict = self.evaluate()

        if figname is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            figfile = f"{figname}_{self.data_fraction:.2f}"
            fig.savefig(self.output_path / f"{figfile}.pdf")

            pd.DataFrame([eval_dict]).to_csv(self.output_path / f"{figfile}.csv")
            # # write JSON files:
            # with (self.output_path / f"{figfile}.json").open("w", encoding="UTF-8") as target:
            #     json.dump(eval_dict, target)

    def evaluate_mean_fun(self):
        covered = 0
        for dp, dt in zip(self.data_post, self.data_true):
            ci = self.calculate_ci(dp.y_cov, dp.y_mean)
            if ci[0] < dt.y < ci[1]:
                covered += 1

        covered_fraction = covered/len(self.data_true)

        return covered_fraction

    def evaluate_overall_mean(self):
        covered = 0
        se_avg = self.se_avg(self.data_post.y_cov)
        ci = self.calculate_ci(se_avg, np.mean(self.data_post.y_mean))
        if ci[0] < np.mean(self.data_true.y) < ci[1]:
            covered = 1

        pred_prob = norm.pdf((np.mean(self.data_true.y) - np.mean(self.data_post.y_mean))/se_avg)
        return {"ci_overall_mean_lb": ci[0], "ci_overall_mean_ub": ci[1],
                "overall_mean_covered": covered, "ci_overall_width": ci[1]-ci[0],
                "pred_prob_overall_mean": pred_prob}

    def evaluate(self):

        covered_fraction = self.evaluate_mean_fun()

        overall_mean = self.evaluate_overall_mean()
        param_error = {k: (v-self.param_sim[k])/abs(self.param_sim[k]) for k, v in self.param_fit.items()}

        pred_logprob_test = self.get_predictive_logprob(self.data_post[self.test_idx],
                                                     self.data_true.y[self.test_idx])

        pred_logprob_train = self.get_predictive_logprob(self.data_post[self.train_idx],
                                                      self.data_true.y[self.train_idx])
        pred_logprob_test_train = self.get_predictive_logprob(self.data_post, self.data_true.y)
        return {"pred_logprob_test": pred_logprob_test, "pred_logprob_train": pred_logprob_train,
                "pred_logprob_test_train": pred_logprob_test_train, **overall_mean,
                "covered_fraction": covered_fraction,
                **param_error}

    def evaluate_multisample(self, n_samples=100):
        gps = copy(self)
        samples = gps.sim_gp(n_samples)
        eval_list = []

        for sample in samples:
            gps.data_true = sample
            try:
                gps.fit(refit=True)
            except Exception as e:
                logger.warning("Could not fit GP")
            eval_list.append(gps.evaluate())
        df = pd.DataFrame(eval_list)
        summary_dict = df.mean(axis=0).to_dict()
        summary_dict["data_fraction"] = gps.data_fraction
        summary_dict["kernel_sim"] = gps.kernel_sim
        summary_dict["n_samples"] = n_samples
        return summary_dict

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

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    data_fraction = 0.1

    kernels_limited = list(OU_KERNELS["bounded"].keys())
    # kernels_limited = [k for k in kernels_limited if k in ["dot"]]
    # data_fraction_weights = get_sesonal_weights(base_config["x"], period=PERIOD_DAY)

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
        performance_summary = []
        for k_name, k in mode_config["kernels"].items():
            if k_name not in kernels_limited:
                continue
            start = datetime.datetime.utcnow()
            logger.info(f"Simulation started for {mode_name}: {k_name}")
            k_norm = GPSimulator.get_normalized_kernel(k)

            gps = GPSimulator(kernel_sim=k_norm, data_fraction=data_fraction, **mode_config["config"])
            gps.plot(figname=f"gp_{k_name}_{mode_name}")
            performance_summary.append(gps.evaluate_multisample(n_samples=100))

    df = pd.DataFrame(performance_summary)
    df.to_csv(OUTPUT_PATH / f"perfomrance_sum_{mode_name}.csv")

        #     output_dict = gps.test_ci(data_fraction=0.3)
        #     ci_info.append({**output_dict, **mode_config["config"]})
        #
        #     logger.info(f"Simulation ended for {mode_name}: {k_name}. "
        #                 f"Duration: {(datetime.datetime.utcnow()-start).total_seconds()} sec")
        #
        # ci_info_df = pd.DataFrame(ci_info)
        # ci_info_df = ci_info_df[[col for col in ci_info_df.columns if col not in ["x", 'mean_f']]]
        # ci_info_df.to_csv(OUTPUT_PATH / f"ci_info_{mode_name}.csv")


