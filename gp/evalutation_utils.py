from scipy.stats import norm
import numpy as np


def mse(true, pred):
    return np.mean((pred - true) ** 2)


def calculate_ci(se, mean, alpha=0.05, dist=norm):
    """
    General method to calculate the confidence/credible intervals from the mean and standard error (se).
    """
    return {"ci_lb": mean - se * dist.ppf(1 - alpha / 2),
            "ci_ub": mean + se * dist.ppf(1 - alpha / 2)}


def se_overall_mean_from_cov(y_cov):
    """
    Standard error of the overall mean of a collection of RV (multivariate normal) with
    Covariance matrix y_cov

    Rational:
    Var(A+B) = Var(A) + Var(B) + 2Cov(A,B)
    Var(c * A) = c^2 * A
    """

    return 1 / y_cov.shape[0] * np.sqrt(np.sum(y_cov))



