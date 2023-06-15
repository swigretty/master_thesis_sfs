import scipy
import numpy as np
from copy import copy
from scipy.stats import norm, multivariate_normal

from gp.gp_regressor import GPData


class GPEvaluator:

    def __init__(self, data_true: GPData, data_post: GPData):
        self.data_true = data_true
        self.data_post = data_post

        assert len(self.data_true) == len(self.data_post)

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

    def get_predictive_logprob(self):
        prob2 = multivariate_normal.logpdf(self.data_true.y, mean=self.data_post.y_mean, cov=self.data_post.y_cov)
        return prob2

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

        c, lower = scipy.linalg.cho_factor(S_fr)

        def solve(B):
            return scipy.linalg.cho_solve((c, lower), B)

        def logdet(S):
            return np.linalg.slogdet(S)[1]

        term1 = np.trace(solve(S_to))
        term2 = logdet(S_fr) - logdet(S_to)
        term3 = d.T @ solve(d)
        return (term1 + term2 + term3 - len(d)) / 2.

    def kl_div_fun(self):
        m_to = self.data_true.y_mean
        S_to = self.data_true.y_cov

        m_fr = self.data_post.y_mean
        S_fr = self.data_post.y_cov
        return self.kl_div((m_to, S_to), (m_fr, S_fr))

    def evaluate_fun(self):
        def calculate_ci_row(row):
            return self.calculate_ci(row["y_var"], row["y_mean"])

        df = self.data_post.to_df()
        df[["ci_lb", "ci_ub"]] = df.apply(lambda row: calculate_ci_row(row), axis=1).to_list()
        df["ci_covered"] = (df["ci_lb"] < self.data_true.y) & (self.data_true.y < df["ci_ub"])

        covered_fraction = np.mean(df["ci_covered"])
        kl_fn = self.kl_div_fun()
        return {"covered_fraction_fun": covered_fraction, "kl_fun": kl_fn, "pred_logprob": self.get_predictive_logprob()}

    def evaluate_overall_mean(self):
        covered = 0
        se_avg = self.se_avg(self.data_post.y_cov)
        ci = self.calculate_ci(se_avg, np.mean(self.data_post.y_mean))
        if ci[0] < np.mean(self.data_true.y) < ci[1]:
            covered = 1

        pred_prob = norm.pdf((np.mean(self.data_true.y) - np.mean(self.data_post.y_mean)) / se_avg)

        m_true = np.mean(self.data_true.y_mean)
        m_post = np.mean(self.data_post.y_mean)
        S_true = self.se_avg(self.data_true.y_cov)**2
        S_post = se_avg**2
        # TODO make this work for 1D sample
        # kl = self.kl_div((m_true, S_true), (m_post, S_post))

        return {"ci_overall_mean_lb": ci[0], "ci_overall_mean_ub": ci[1],
                "overall_mean_covered": covered, "ci_overall_width": ci[1]-ci[0],
                "pred_prob_overall_mean": pred_prob}

    def evaluate(self):
        eval_fun_dict = self.evaluate_fun()
        overall_mean = self.evaluate_overall_mean()
        return {**overall_mean, **eval_fun_dict}








