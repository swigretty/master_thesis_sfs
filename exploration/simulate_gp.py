from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from dataclasses import dataclass
from functools import lru_cache
from copy import copy
from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
from matplotlib.colors import CSS4_COLORS
import matplotlib as mpl
from scipy.stats import norm

from exploration.gp import GPModel, plot_gpr_samples, plot_kernel_function, plot_posterior
from exploration.constants import PLOT_PATH
from exploration.explore import get_red_idx
from log_setup import setup_logging

logger = getLogger(__name__)

mpl.style.use('seaborn-v0_8')

@dataclass
class GPData():
    x : np.array
    y : np.array
    n : int
    y_mean : np.array = None
    y_cov : np.array = None


class GPSimulator():

    def __init__(self, x=np.linspace(0, 40, 200), kernel_sim=1 * Matern(nu=0.5, length_scale=1), mean_f=lambda x: 120,
                 meas_noise=0, kernel_fit=None):

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        self.x = x
        self.kernel_sim = kernel_sim

        self.mean_f = mean_f
        self.meas_noise = meas_noise
        self.offset = mean_f(0)

        if kernel_fit is None:
            kernel_fit = kernel_sim + ConstantKernel(constant_value=self.offset ** 2, constant_value_bounds="fixed")
        self.kernel_fit = kernel_fit

        self.gpm_sim = GPModel(kernel=self.kernel_sim, normalize_y=False)
        self.gpm_fit = GPModel(kernel=self.kernel_fit, normalize_y=False)

        logger.info(f"Initialized {self.__class__.__name__} with \n {kernel_sim=} \n {kernel_fit=}")

    # @lru_cache()
    # @property
    # def data_prior(self):
    #     return self.sim_gp()
    #
    def sim_gp(self, n_samples=5):
        y_prior, y_prior_mean, y_prior_cov = self.gpm_sim.sample_from_prior(
            self.x, n_samples=n_samples, mean_f=self.mean_f)

        # if self.meas_noise:
        #     y_prior = y_prior + self.meas_noise * np.random.standard_normal((y_prior.shape))
        #     y_prior_cov[np.diag_indices_from(y_prior_cov)] += meas_noise
        return GPData(x=self.x, y=y_prior, y_mean=y_prior_mean, y_cov=y_prior_cov, n=len(x))

    def subsample_data_sim(self, data_sim, data_fraction=0.3, data_fraction_weights=None):
        idx = get_red_idx(data_sim.n, data_fraction=data_fraction, weights=data_fraction_weights)
        y_red = data_sim.y[idx, ]
        x_red = data_sim.x[idx]

        data_sim_sub = GPData(x=x_red, y=y_red, n=len(x))
        return data_sim_sub

    def plot_prior(self, ax, data_prior):
        plot_lim = 30
        y_prior_std = np.diag(data_prior.y_cov)
        ylim = None
        if any(y_prior_std) > plot_lim:
            ylim = [np.mean(data_prior.y) - plot_lim, np.mean(data_prior.y) + plot_lim]
        plot_gpr_samples(ax, data_prior.x, data_prior.y_mean, y_prior_std, y=data_prior.y, ylim=ylim)
        ax.set_title("Samples from prior distribution")

    def choose_sample_from_prior(self, data_prior, data_index: int = 0):
        data_single = copy(data_prior)
        data_single.y = data_prior.y[:, data_index]
        return data_single

    def fit(self, data):
        self.gpm_fit.fit(data.x, data.y)

    def plot_posterior(self, ax, data, y_true=None):
        y_post, y_post_mean, y_post_cov = self.gpm_fit.sample_from_posterior(self.x, n_samples=1)
        plot_posterior(ax, self.x, y_post_mean, np.diag(y_post_cov), data.x, data.y, y_true=y_true)

    def sim_fit_plot(self, data_fraction_list=np.logspace(-2, 0, 5), figname=None):
        data_prior = self.sim_gp()
        data_true = self.choose_sample_from_prior(data_prior, data_index=0)

        fig_list = []
        for data_fraction in data_fraction_list:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 15))
            self.plot_prior(ax[0, 0], data_prior)
            plot_kernel_function(ax[0, 1], data_true.x, self.kernel_sim)

            data = self.subsample_data_sim(data_true, data_fraction=data_fraction)

            self.fit(data)
            self.plot_posterior(ax[1, 0], data, y_true=data_true.y)

            plot_kernel_function(ax[1, 1], data_true.x, self.gpm_fit.gp.kernel_)

            if figname is not None:
                fig.savefig(PLOT_PATH / f"{figname}_{data_fraction:.2f}.pdf")
            fig_list.append(fig)
        return fig

    @staticmethod
    def calculate_ci(se, mean, alpha=0.05, dist=norm):
        return (mean - se * dist.ppf(1 - alpha/2), mean + se * dist.ppf(1 - alpha/2))

    @staticmethod
    def se_avg(y_post_cov):
        return 1 / y_post_cov.shape[0] * np.sqrt(np.sum(y_post_cov))

    def test_ci(self, n_samples=100, data_fraction=0.3):
        """
        simulate from the prior,
        then simulate from the model using those values from the prior, and
        estimate the parameters using the same prior.
        """
        data_prior = self.sim_gp(n_samples=n_samples)
        n_ci_coverage = 0
        for sample_index in range(n_samples):
            data_true = self.choose_sample_from_prior(data_prior, data_index=sample_index)
            data = self.subsample_data_sim(data_true, data_fraction=data_fraction)
            self.fit(data)
            y_post_mean, y_post_cov = self.gpm_fit.predict(self.x, return_cov=True)
            ci = self.calculate_ci(self.se_avg(y_post_cov), np.mean(y_post_mean))
            if ci[0] < np.mean(data_true.y) < ci[1]:
                n_ci_coverage += 1
        ci_coverage = n_ci_coverage/n_samples
        logger.info(f"{ci_coverage=} with {data_fraction=}")


def mean_fun_const(x):
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    return 120



if __name__ == "__main__":
    setup_logging()

    # measuring time in hours
    n_days = 3
    samples_per_hour = 10
    period_day = 24
    period_week = 7 * period_day
    x = np.linspace(0, period_day * n_days, period_day * n_days * samples_per_hour)

    # Simple Kernels
    constant_kernel = ConstantKernel(constant_value=0, constant_value_bounds="fixed")
    ar1_kernel = ConstantKernel(constant_value=1, constant_value_bounds="fixed") * Matern(
        nu=0.5, length_scale=3, length_scale_bounds="fixed")
    long_term_trend_kernel = ConstantKernel(constant_value=4, constant_value_bounds="fixed") * RBF(
        length_scale=50, length_scale_bounds="fixed")
    short_term_trend_kernel = ConstantKernel(constant_value=4, constant_value_bounds="fixed") * RBF(
        length_scale=3, length_scale_bounds="fixed")
    short_cycle_kernel = ConstantKernel(constant_value=10, constant_value_bounds="fixed") * ExpSineSquared(
        length_scale=3, periodicity=period_day, periodicity_bounds="fixed", length_scale_bounds="fixed")
    long_cycle_kernel = ConstantKernel(constant_value=10, constant_value_bounds="fixed") * ExpSineSquared(
        length_scale=3, periodicity=period_week, periodicity_bounds="fixed", length_scale_bounds="fixed")
    kernel_dot = DotProduct(sigma_0=1, sigma_0_bounds="fixed")

    kernels = {"white": WhiteKernel(noise_level=1, noise_level_bounds="fixed"),
        "ar1": ar1_kernel, "dot": ar1_kernel + kernel_dot, "rbf": ar1_kernel + long_term_trend_kernel,
               "sin": ar1_kernel + short_cycle_kernel,
               "sinrbf": ar1_kernel + short_cycle_kernel * long_term_trend_kernel}

    base_config = dict(
        x=x,
        meas_noise=0,
        mean_f=mean_fun_const
    )

    for k_name, k in kernels.items():
        gps = GPSimulator(kernel_sim=k, **base_config)
        # gps.sim_fit_plot(figname=f"sim_fit_gp_{k_name}")
        gps.test_ci(data_fraction=0.8)








