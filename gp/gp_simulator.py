import datetime
import inspect
from functools import partial
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
import logging
import json
import scipy
import matplotlib as mpl
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, Product
from gp.gp_regressor import GPR
from gp.gp_plotting_utils import plot_kernel_function, plot_posterior, plot_gpr_samples, Plotter
from gp.gp_data import GPData
from gp.baseline_methods import BASELINE_METHODS
from gp.target_measures import TARGET_MEASURES, ci_overall_mean_gp
from constants.constants import get_output_path
from exploration.explore import get_red_idx
from gp.simulate_gp_config import base_config, OU_KERNELS, PARAM_NAMES
from gp.evaluate import GPEvaluator, SimpleEvaluator
from log_setup import setup_logging

logger = getLogger(__name__)

# mpl.style.use('seaborn-v0_8')


def weights_func_value(y):
    return y - np.min(y)


class GPSimulator():

    def __init__(self, x=np.linspace(0, 40, 200), kernel_sim=1 * Matern(nu=0.5, length_scale=1), mean_f=lambda x: 120,
                 meas_noise_var=0, kernel_fit=None, normalize_y=False, output_path=get_output_path,
                 data_fraction_weights=None, data_fraction=0.3, f_true=None, meas_noise=None, rng=None,
                 normalize_kernel=False, scale=1):

        self.sim_time = datetime.datetime.utcnow()

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.x = x

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.mean_f = mean_f
        self.meas_noise_var = meas_noise_var
        self.offset = np.mean([mean_f(xi) for xi in self.x])
        self.scale = scale

        self.data_fraction_weights = data_fraction_weights
        self.data_fraction = data_fraction

        self.kernel_sim = kernel_sim
        self.normalize_kernel = normalize_kernel
        if normalize_kernel:
            self.kernel_sim, self.meas_noise_var, self.scale = self.get_normalized_kernel(kernel_sim,
                                                                                     meas_noise_var=self.meas_noise_var)

        if kernel_fit is None:
            kernel_fit = self.kernel_sim

        self.kernel_fit = kernel_fit
        self.normalize_y = normalize_y
        self.gpm_sim = GPR(kernel=self.kernel_sim, normalize_y=False, optimizer=None, rng=rng,
                           alpha=0)  # No Variance for fitting the gpm_sim and getting true decomposed

        self.f_true = f_true  # Noise free BP values, if None, samples from self.gpm_sim
        # Vector of measurement noise values for every input in x.
        # If None, draws iid samples from normal dist with var=self.meas_noise_var
        self.meas_noise = meas_noise
        # Fit the true GP once. This is needed for the true mean decomposition
        self.gpm_sim.fit(self.y_true.x, self.f_true.y)

        self.meas_noise_var_fit = self.meas_noise_var
        if self.normalize_y:
            self._y_train_std = np.std(self.y_true_train.y)
            self.meas_noise_var_fit = self.meas_noise_var_fit / self._y_train_std**2
        self.gpm_fit = GPR(kernel=self.kernel_fit, normalize_y=self.normalize_y, alpha=self.meas_noise_var_fit, rng=rng)

        self.output_path = output_path
        if self.output_path is not None:
            if callable(self.output_path):
                self.output_path = self.output_path(self.sim_time)
            self.output_path.mkdir(parents=True)
            with (self.output_path/"config.json").open("w") as f:
                f.write(json.dumps(self.current_config, default=str))
            self.df.to_csv(self.output_path / f"data.csv")

        init_kwargs = {k: v for k, v in self.current_init_kwargs.items() if not isinstance(v, np.ndarray)}
        logger.info(f"Initialized {self.__class__.__name__} with: \n {init_kwargs=}")

    @property
    def df(self):
        is_training_data = [idx in self.train_idx for idx in range(len(self.x))]
        return pd.DataFrame({"x": self.x.reshape(-1), "f_true": self.f_true.y, "meas_noise": self.meas_noise,
                             "training_data": is_training_data})

    def sim_gp(self, n_samples=5, predict_y=False):

        # Predict noise free in any case, since gpm_sim will be initialized with alpha=0 and
        # measurment noise later if predict_y=True
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(self.x, n_samples=n_samples,
                                                                            predict_y=False)
        if predict_y:
            # samples with noise
            y_prior_noisy = y_prior + np.sqrt(self.meas_noise_var) * self.rng.standard_normal((y_prior.shape))
            y_prior_cov_noisy = y_prior_cov + np.diag(np.repeat(self.meas_noise_var, len(self.x)))
            data = [GPData(x=self.x, y=y_prior_noisy[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov_noisy) for idx in
                    range(n_samples)]
        else:
            data = [GPData(x=self.x, y=y_prior[:, idx], y_mean=y_prior_mean, y_cov=y_prior_cov) for idx in
                    range(n_samples)]

        return data

    def get_seasonality(self):
        true_dec = self.gpm_sim.predict_mean_decomposed(self.x)
        fun = [fun for name, fun in true_dec.items() if "ExpSineSquared" in name]
        assert len(fun) == 1, "cannot extract seasonal pattern"
        weights = fun[0] - min(fun[0])
        return weights

    # @cached_property
    # def f_true_post(self):
    #     """
    #     this is simply f_true with Cov = 0 everywhere
    #     """
    #     self.gpm_sim.fit(self.f_true.x, self.f_true.y)
    #     f_post_mean, f_post_cov = self.gpm_sim.predict(self.y_true.x, return_cov=True)
    #     assert np.mean((self.f_true.y - f_post_mean) ** 2) < 10 ** (-4)
    #     return GPData(x=self.f_true.x, y_mean=f_post_mean, y_cov=f_post_cov)

    @property
    def meas_noise(self):
        return self._meas_noise

    @meas_noise.setter
    def meas_noise(self, data=None):
        self._meas_noise = data
        if data is None:
            self._meas_noise = np.sqrt(self.meas_noise_var) * self.rng.standard_normal(len(self.x))

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
        if hasattr(self, "_f_true"):
            raise Exception("f_true is already set")
        self._f_true = data
        if data is None:
            self._f_true = self.sim_gp(n_samples=1, predict_y=False)[0]

    @property
    def train_idx(self):
        if not hasattr(self, "_train_idx"):
            if callable(self.data_fraction_weights):
                weights = self.data_fraction_weights(self.get_seasonality())
            elif self.data_fraction_weights == "seasonal":
                weights = self.get_seasonality()
            else:
                weights = self.data_fraction_weights

            self._train_idx = get_red_idx(len(self.x), data_fraction=self.data_fraction, weights=weights, rng=self.rng)

        return self._train_idx

    @property
    def test_idx(self):
        """
        this is very slow [idx for idx in range(len(self.x)) if idx not in self.train_idx]
        """
        if not hasattr(self, "_test_idx"):
            self._test_idx = np.setdiff1d(np.arange(len(self.x)), self.train_idx)
        return self._test_idx

    @property
    def y_true_train(self):
        """
        This represents the (potentially noisy) subsampled measurements, used to fit the GP
        """
        if not hasattr(self, "_y_true_train"):
            self._y_true_train = self.y_true[self.train_idx]
        return self._y_true_train

    @property
    def y_true_samples(self):
        if not hasattr(self, "_y_true_samples"):
            self._y_true_samples = self.sim_gp(n_samples=5, predict_y=True) + [self.y_true]
        return self._y_true_samples

    @property
    def param_sim(self):
        return self.extract_params_from_kernel(self.kernel_sim)

    @property
    def param_fit(self):
        if not hasattr(self.gpm_fit, "X_train_"):
            raise "GP has not been fitted yet"
        return self.extract_params_from_kernel(self.gpm_fit.kernel_)

    @property
    def f_post(self):
        """
        The posterior distribution over f based on the fitted gaussian process model
        """
        if not hasattr(self.gpm_fit, "X_train_"):
            raise "GP has not been fitted yet"
        if not hasattr(self, "_f_post"):
            f_post_mean, f_post_cov = self.gpm_fit.predict(self.x, return_cov=True)
            self._f_post = GPData(x=self.x, y_mean=f_post_mean, y_cov=f_post_cov)
        return self._f_post

    @property
    def pred_empirical_mean(self):
        """
        Using the overall mean as prediction.
        cov is calculated assuming iid data.
        """
        if not hasattr(self, "_pred_empirical_mean"):
            sigma_mean = 1 / len(self.y_true_train.y) * np.var(self.y_true_train.y)
            y_cov = np.zeros((len(self.x), len(self.x)), float)
            np.fill_diagonal(y_cov, sigma_mean)
            y = np.repeat(np.mean(self.y_true_train.y), len(self.x))
            self._pred_empirical_mean = GPData(x=self.x, y_mean=y, y_cov=y_cov)
        return self._pred_empirical_mean

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

    @staticmethod
    def get_normalized_kernel(kernel, meas_noise_var=0):
        previousloglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)

        std_range = (0.99, 1.01)
        i = 0
        scale = 1
        y_std = 2

        while (std_range[0] > y_std) or (
                std_range[1] < y_std):

            kernel_ = ConstantKernel(constant_value=1/scale**2, constant_value_bounds="fixed") * kernel
            gps = GPSimulator(kernel_sim=kernel_, meas_noise_var=meas_noise_var / scale ** 2,
                              rng=np.random.default_rng(15), output_path=None, normalize_kernel=False)
            data_sim = gps.sim_gp(n_samples=1000, predict_y=True)
            y_std = np.mean([np.std(d.y) for d in data_sim])
            scale *= y_std
            i += 1
            if i > 100:
                break

        logger.setLevel(previousloglevel)

        logger.info(f"final kernel {kernel_} with scaling {1/scale} and {y_std=}")

        return kernel_, meas_noise_var / scale ** 2, scale

    def choose_sample_from_prior(self, data_index: int = 0):
        data_single = copy(self.y_true_samples)
        data_single.y = self.y_true_samples.y[:, data_index]
        return data_single

    def fit(self, refit=False):
        if (not hasattr(self.gpm_fit, "X_train_")) or refit:
            self.gpm_fit.fit(self.y_true_train.x, self.y_true_train.y)

        if self.normalize_y:
            assert (self._y_train_std - self.gpm_fit._y_train_std) < 0.0000001

    @Plotter
    def plot_prior(self, title="Samples from Prior Distribution", ax=None):
        data_prior = self.y_true_samples

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
    def plot_posterior(self, title="Predictive Distribution", ax=None, pred_data=None,
                       y_true_subsampled=None, **kwargs):
        if pred_data is None:
            pred_data = self.f_post
        if y_true_subsampled is None:
            y_true_subsampled = self.y_true_train
        data_dict = {"f_post": pred_data, "y_true_subsampled": y_true_subsampled,
                     "f_true": self.f_true}

        plot_posterior(self.x, data_dict["f_post"].y_mean, y_post_std=data_dict["f_post"].y_std,
                       x_red=data_dict["y_true_subsampled"].x, y_red=data_dict["y_true_subsampled"].y,
                       y_true=data_dict["f_true"].y, ax=ax)
        ax.set_title(title)

    @Plotter
    def plot_true_with_samples(self, ax=None):
        f_true = self.f_true
        y_true = self.y_true
        data = self.y_true_train

        ax.set_title("True Time Series with Samples")
        ax.plot(f_true.x, f_true.y, "r:",
                label=f"var(f_true): {np.var(f_true.y)}, var(y_true): {np.var(y_true.y)}")
        ax.scatter(data.x, data.y, color="red", zorder=5, label=f"{np.var(data.y)=}")
        ax.legend()

    def plot(self):
        nrows = 3
        ncols = 2

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 6))

        self.plot_prior(ax=ax[0, 0])

        plot_kernel_function(self.x, self.kernel_sim, ax=ax[0, 1])

        self.fit()

        self.plot_posterior(ax=ax[1, 0])

        plot_kernel_function(self.x, self.gpm_fit.kernel_, ax=ax[1, 1])

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            ax[2, 0].plot(self.x, v, label=k)
        ax[2, 0].legend()

        for k, v in self.gpm_fit.predict_mean_decomposed(self.x).items():
            ax[2, 1].plot(self.x, v, label=k)

        fig.tight_layout()
        figfile = "fit"
        if self.output_path:
            fig.savefig(self.output_path / f"{figfile}.pdf")
            plt.close(fig)
        return fig

    def plot_errors(self):
        nrows = 2
        ncols = 2
        current_row = 0
        current_col = 0
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*10, ncols*10))
        self.plot_posterior(ax=axs[current_row, current_col])

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
        figfile = "err"
        if self.output_path:
            fig.savefig(self.output_path / f"{figfile}.pdf")
            plt.close(fig)

    @property
    def eval_config(self):
        eval_config = {"train_perf": {"idx": self.train_idx, "fun": "evaluate_fun"},
                       "test_perf": {"idx": self.test_idx, "fun": "evaluate_fun"},
                       "overall_perf": {"idx": None, "fun": "evaluate"}}
        return eval_config

    @property
    def eval_data(self):
        return {"y_true": self.y_true.y, "f_true": self.f_true, "f_pred": self.f_post}

    def get_decomposed_variance(self):
        variance_out = {}

        for k, v in self.gpm_sim.predict_mean_decomposed(self.x).items():
            variance_out[k] = np.var(v)
            if "Sine" in k:
                variance_out["ampl"] = np.sqrt(2 * variance_out[k])
                variance_out["dip_ampl"] = variance_out["ampl"] * 2/np.pi

        variance_out["f_true"] = np.var(self.f_true.y)
        variance_out["y_true"] = np.var(self.y_true.y)
        variance_out["y_true_train"] = np.var(self.y_true_train.y)

        variance_out["y_true_std"] = np.sqrt(variance_out["y_true"])
        variance_out["y_true_train_std"] = np.sqrt(variance_out["y_true_train"])

        return variance_out

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

        eval_df = pd.DataFrame([output_dict])
        if self.output_path:
            eval_df.to_csv(self.output_path / f"evaluate.csv")
        return output_dict

    @property
    def current_init_kwargs(self):
        return {k: v for k, v in vars(self).items() if k in inspect.signature(GPSimulator.__init__).parameters.keys()}

    @property
    def current_config(self):
        base = {k: v for k, v in self.current_init_kwargs.items() if not isinstance(v, np.ndarray)}
        base["y_train_std"] = self._y_train_std
        return base

    @Plotter
    def mean_decomposition_plot(self, ax=None):
        decomposed_dict_sim = self.gpm_sim.predict_mean_decomposed(self.x)
        ax.plot(self.f_true.x, self.f_true.y)
        sum_of_v = np.zeros(len(self.x))
        for k, v in decomposed_dict_sim.items():
            ax.plot(self.x, v, label=k, linestyle="dashed")
            sum_of_v += v

        assert np.all(sum_of_v - self.f_true.y < 0.00000001)
        # ax.plot(data.x, sum_of_v, label="sum")
        # ax.plot(data.x, y_post_mean, label="post_mean")
        ax.legend()


class GPSimulationEvaluator(GPSimulator):

    def __init__(self,  kernel_sim=None, baseline_methods=None, normalize_kernel=True, meas_noise_var=0,
                 target_measures=None, **gps_kwargs):

        self.kernel_sim_orig = kernel_sim
        self.meas_noise_var_orig = meas_noise_var
        self.gps_kwargs_orig = {"kernel_sim": self.kernel_sim_orig, "meas_noise_var": self.meas_noise_var_orig,
                                "normalize_kernel": normalize_kernel, **gps_kwargs}

        super().__init__(**self.gps_kwargs_orig)

        self.gps_kwargs_normalized = {"kernel_sim": self.kernel_sim, "meas_noise_var": self.meas_noise_var,
                                      "normalize_kernel": False, **gps_kwargs}

        if baseline_methods is None:
            baseline_methods = BASELINE_METHODS
        self.baseline_methods = copy(baseline_methods)
        if target_measures is None:
            target_measures = TARGET_MEASURES
        self.target_measures = target_measures

    def _get_pred_method(self, method, **kwargs):
        assert all(self.y_true_train.x == sorted(self.y_true_train.x))
        return method(self.x, self.y_true_train.x, self.y_true_train.y, **kwargs)

    def _get_pred_baseline(self, **kwargs):
        """
        returns {"data": predicted_data, "ci_fun": function_to_calculate_ci}
        """
        pred_baseline = {}
        for eval_name, eval_fun in self.baseline_methods.items():
            pred_baseline[eval_name] = self._get_pred_method(eval_fun, **kwargs)
            if fun_new := pred_baseline[eval_name].get("fun"):
                # overwrite self.baseline_method e.g. for spline smoothing parameter has been found using cv
                self.baseline_methods[eval_name] = fun_new

        return pred_baseline

    @property
    def pred_baseline(self):
        if not hasattr(self, "_pred_baseline"):
            self._pred_baseline = self._get_pred_baseline()
        return self._pred_baseline

    @property
    def predictions(self):
        return {"gp": {"data": self.f_post, "ci_overall_mean": ci_overall_mean_gp(
            self.f_post.y_mean, y_cov=self.f_post.y_cov), "ci_fun": self.target_measures_from_posterior},
                **self.pred_baseline}

    def target_measures_from_posterior(self, theta_fun=None, n_samples=100, alpha=0.05, **kwargs):
        if theta_fun is None:
            theta_fun = self.target_measures
        if not isinstance(theta_fun, list):
            theta_fun = [theta_fun]
        # posterior_samples, y_mean, y_cov = self.gpm_fit.sample_from_posterior(self.x, n_samples=n_samples)
        out_dict = {}
        posterior_samples = self.rng.multivariate_normal(self.f_post.y_mean, self.f_post.y_cov, n_samples).T
        for target_measure in theta_fun:
            target_measure_samples = np.apply_along_axis(target_measure, 0, posterior_samples)
            theta_hat = np.mean(target_measure_samples)

            out_dict[target_measure.__name__] = {
                "mean": theta_hat, "ci_lb": np.quantile(target_measure_samples, alpha),
                "ci_ub": np.quantile(target_measure_samples, 1-alpha)}

        return out_dict

    @property
    def true_measures(self):
        if not hasattr(self, "_true_measures"):
            self._true_measures = {m.__name__: m(self.f_true.y) for m in self.target_measures}
        return self._true_measures

    def evaluate_target_measures(self, ci_fun_kwargs=None):
        eval_output = []
        if ci_fun_kwargs is None:
            ci_fun_kwargs = {}
        ci_fun_kwargs["theta_fun"] = self.target_measures
        for method_name, pred in self.predictions.items():
            logger.info(f"Evaluate {method_name=}")
            pred_measures = pred["ci_fun"](**ci_fun_kwargs)
            for measure in self.target_measures:
                pred_m = pred_measures[measure.__name__]
                # TODO write test for this, this does not belong here
                # if ci2 := pred.get(f"ci_{measure.__name__}"):
                #     logger.info(f"{pred_m}, {ci2=}")

                eval = SimpleEvaluator(f_true=self.true_measures[measure.__name__],
                                       f_pred=pred_m["mean"], ci_lb=pred_m["ci_lb"], ci_ub=pred_m["ci_ub"])

                eval_output.append({"method": method_name, "target_measure": measure.__name__, **eval.to_dict()})

        if self.output_path:
            pd.DataFrame(eval_output).to_csv(self.output_path / f"evaluate_target_measures.csv")
        return eval_output

    def plot_posterior_baseline(self):
        for method, pred in self.pred_baseline.items():
            self.plot_posterior(pred_data=pred["data"], title=f"Prediction {method}", figname_suffix=f"{method}")

    def plot(self):
        super().plot()
        self.plot_posterior_baseline()

    def evaluate_multisample(self, n_samples=100, only_var=False):
        current_config = copy(self.gps_kwargs_normalized)
        current_config["output_path"] = None
        # TODO is it fair to reuse baseline methods?
        # use the baseline method just identified, so you don't have to reevaluate smoothing params for spline
        # each time with CV
        current_config["baseline_methods"] = self.baseline_methods
        current_config["target_measures"] = self.target_measures

        eval_dict = {}
        eval_target_measure = []
        variances = []

        previousloglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)
        for i in range(n_samples):
            gps = GPSimulationEvaluator(**current_config)

            if not only_var:
                gps.fit()
                # TODO eval_dict as df with group by and aggregate
                eval_sample = gps.evaluate()
                eval_dict = {k: [v] + eval_dict.get(k, []) for k, v in eval_sample.items()}
                eval_target_measure.extend(gps.evaluate_target_measures(ci_fun_kwargs={"logger": logger}))
            variances.append(gps.get_decomposed_variance())

        logger.setLevel(previousloglevel)

        variance_df = pd.DataFrame(variances)
        variance_df.describe().to_csv(self.output_path / f"variance_summary.csv")

        for col in variance_df.columns:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.hist(variance_df[col], bins=max(int(n_samples/4), 1))
            ax.axvline(np.mean(variance_df[col]), color='k', linestyle='dashed', linewidth=1)
            fig.savefig(self.output_path / f"variance_{col}_summary.pdf")
            plt.close(fig)

        if eval_dict:
            eval_dict = {k: pd.DataFrame(v).mean(axis=0).to_dict() for k, v in eval_dict.items()}
            for v in eval_dict.values():
                v["data_fraction"] = gps.data_fraction
                v["kernel_sim"] = gps.kernel_sim
                v["kernel_fit"] = gps.gpm_fit.kernel_
                v["n_samples"] = n_samples
                v["meas_noise_var"] = gps.meas_noise_var
                v["output_path"] = self.output_path
            with (self.output_path / "eval_summary.json").open("w") as f:
                f.write(json.dumps(eval_dict, default=str))

        if eval_target_measure:
            eval_target_measure = pd.DataFrame(eval_target_measure).groupby(["method", "target_measure"]).agg(
                "mean").reset_index(drop=False)
            eval_target_measure["n_samples"] = n_samples
            eval_target_measure.to_csv(self.output_path / "eval_measure_summary.csv")
        plt.close("all")
        return eval_dict, eval_target_measure

    # @Plotter
    # def plot_overall_mean(self, ax=None, gps=None):
    #     if gps is None:
    #         gps = self
    #     if gps is self:
    #         pred_baseline = self.pred_baseline
    #     else:
    #         pred_baseline = self._get_pred_baseline(gps)
    #
    #     gps.plot_true_with_samples(ax=ax, add_offset=True)
    #     plot_lines_dict = {"true_mean": gps.f_true.y,
    #                        "gp_mean": gps.f_post.y_mean}
    #
    #     for k, v in pred_baseline.items():
    #         plot_lines_dict[f"{k}_mean"] = v["data"].y_mean
    #
    #     # loosely dashed, dashed dotted, dotted
    #     linestyles = [(0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 10))]
    #     for i, (k, v) in enumerate(plot_lines_dict.items()):
    #         ax.plot(gps.x, np.repeat(np.mean(v) + gps.offset, len(gps.x)), label=k, linestyle=linestyles[i])
    #     ax.legend()

    def plot_gp_regression_sample(self, nplots=1, plot_method=None):
        gps_kwargs = self.gps_kwargs_normalized
        gps = self
        for i in range(nplots):
            if nplots > 1:
                gps_kwargs["rng"] = np.random.default_rng(i)
                gps = GPSimulationEvaluator(**gps_kwargs)
            if plot_method is None:
                gps.plot_true_with_samples()
                gps.plot()
                gps.plot_errors()
            else:
                getattr(gps, plot_method)()
            # self.plot_overall_mean(gps=gps)
        return

    # def bootstrap(self, pred_fun, theta_fun, train_x=None, n_samples=100, alpha=0.05):
    #     thetas = {}
    #     train_y = self.y_true_train.y
    #     if train_x is None:
    #         train_x = self.y_true_train.x
    #     if not isinstance(theta_fun, list):
    #         theta_fun = theta_fun
    #
    #     for i in range(n_samples):
    #         idx = sorted(self.rng.choice(np.arange(len(train_y)), size=len(train_y), replace=True))
    #         y_sub = train_y[idx]
    #         x_sub = train_x[idx]
    #         pred = pred_fun(self.x, x_sub, y_sub)
    #         if i == 0:
    #             fun = pred_fun
    #             if isinstance(pred_fun, partial):
    #                 fun = pred_fun.func
    #             self.plot_posterior(pred_data=pred["data"], y_true_subsampled=GPData(x=x_sub, y=y_sub),
    #                                 title=f"Prediction {fun.__name__}",
    #                                 figname_suffix=f"bootstrap_{fun.__name__}")
    #         for fun in theta_fun:
    #             theta = fun(pred["data"].y_mean)
    #             thetas_list = thetas.get(fun, [])
    #             thetas_list.append(theta)
    #
    #     theta_hat = {k: np.mean(v) for k, v in thetas.items()}
    #     ci = {k: {"mean": theta_hat[k], "ci_lb": 2*theta_hat[k]-np.quantile(v, 1-(alpha/2)),
    #           "ci_ub": 2*theta_hat[k]-np.quantile(v, (alpha/2))} for k, v in thetas.items()}
    #     return ci


def plot_mean_decompose(kernel="sin_rbf"):
    gpm = GPSimulator(kernel_sim=OU_KERNELS["fixed"][kernel], **base_config)
    gpm.mean_decomposition_plot(figname=kernel)


if __name__ == "__main__":
    setup_logging()

