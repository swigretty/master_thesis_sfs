import scipy
import numpy as np
from copy import copy
from scipy.stats import norm, multivariate_normal
from gp.evalutation_utils import calculate_ci, se_overall_mean_from_cov
import matplotlib.pyplot as plt
from gp.gp_data import GPData
from gp.gp_plotting_utils import plot_posterior
from logging import getLogger


logger = getLogger(__name__)


class GPEvaluator:

    def __init__(self, y_true: np.array, signal_true: GPData, data_post: GPData, meas_noise: float = 0.0):
        """

        y_true: The noisy measurements i.e. signal_true.y_mean + np.norm(mean=0, var=meas_noise)
        signal_true: The true distribution over the signal f(x). The true GP is completely characterized by
        the signal_true.y_mean and signal_true.y_cov. However, y_true is a realization of the true GP.
        data_post: The predicted distribution over the signal f(x). It is completely characterized by data_post.y_mean
        and data_post.y_cov

        """
        self.y_true = y_true
        self.signal_true = signal_true
        self.data_post = data_post
        self.meas_noise = meas_noise

        self.data_post_y = GPData(x=self.data_post.x, y_mean=self.data_post.y_mean,
                                  y_cov=self.data_post.y_cov + np.diag(np.repeat(self.meas_noise,
                                                                                 len(self.data_post))))

        assert len(self.signal_true) == len(self.data_post) == len(self.data_post_y)

    def get_predictive_logprob_y(self):
        prob2 = multivariate_normal.logpdf(self.y_true, mean=self.data_post_y.y_mean, cov=self.data_post_y.y_cov)
        return prob2

    def mse(self):
        return np.square(self.y_true - self.data_post_y.y_mean).mean()

    @staticmethod
    def kl_div(to, fr):
        """
        Calculate KL divergence, `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices

         simple interpretation of the KL divergence of "to" from "fr" is the expected excess surprise from using
          "fr" as a model when the actual distribution is "to".
        """
        m_to, S_to = to
        m_fr, S_fr = fr

        d = m_fr - m_to

        try:
            c, lower = scipy.linalg.cho_factor(S_fr)
        except Exception as e:
            logger.warning(e)
            return

        def solve(B):
            return scipy.linalg.cho_solve((c, lower), B)

        def logdet(S):
            return np.linalg.slogdet(S)[1]

        term1 = np.trace(solve(S_to))
        term2 = logdet(S_fr) - logdet(S_to)
        term3 = d.T @ solve(d)
        return (term1 + term2 + term3 - len(d)) / 2.

    def kl_div_fun(self):
        m_to = self.signal_true.y_mean
        S_to = self.signal_true.y_cov

        m_fr = self.data_post.y_mean
        S_fr = self.data_post.y_cov
        return self.kl_div((m_to, S_to), (m_fr, S_fr))

    @property
    def ci_covered_meanfun(self):
        return (self.data_post.ci["ci_lb"] < self.signal_true.y_mean) & (
                self.signal_true.y_mean < self.data_post.ci["ci_ub"])

    @property
    def ci_covered_yfun(self):
        return (self.data_post_y.ci["ci_lb"] < self.y_true) & (
                self.y_true < self.data_post_y.ci["ci_ub"])

    def evaluate_fun(self):
        return {"covered_fraction_fun": np.mean(self.ci_covered_meanfun),
                "covered_fraction_yfun": np.mean(self.ci_covered_yfun), "kl_fun": self.kl_div_fun(),
                "pred_logprob": self.get_predictive_logprob_y(), "mse": self.mse()}

    def evaluate_overall_mean(self):
        covered = 0
        se_pred = se_overall_mean_from_cov(self.data_post.y_cov)
        mean_pred = np.mean(self.data_post.y_mean)
        mean_true = np.mean(self.signal_true.y_mean)
        ci = calculate_ci(se_pred, mean_pred)
        if ci[0] < mean_true < ci[1]:
            covered = 1

        pred_prob = norm.pdf((mean_true - mean_pred) / se_pred)

        # TODO make this work for 1D sample
        # S_true = se_overall_mean_from_cov(self.data_post.y_cov)**2
        # S_post = se_avg**2
        # kl = self.kl_div((m_true, S_true), (m_post, S_post))

        return {"ci_overall_mean_lb": ci[0], "ci_overall_mean_ub": ci[1],
                "overall_mean_covered": covered, "ci_overall_width": ci[1]-ci[0],
                "pred_prob_overall_mean": pred_prob, "mse_overll_mean": (mean_true-mean_pred)**2}

    def evaluate(self):
        eval_fun_dict = self.evaluate_fun()
        overall_mean = self.evaluate_overall_mean()
        return {**overall_mean, **eval_fun_dict}

    def plot_errors(self, ax=None):
        error_data_true = GPData(np.array([]))
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        error_idx = np.nonzero(self.ci_covered_meanfun == 0)[0]
        if len(error_idx) > 0:
            error_data_true = self.signal_true[error_idx]

        plot_posterior(self.data_post.x, self.data_post.y_mean, y_post_std=self.data_post.y_std,
                       x_red=error_data_true.x, y_red=error_data_true.y_mean,
                       y_true=self.signal_true.y_mean, ax=ax)







