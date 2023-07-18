import numpy as np
from scipy.stats.distributions import norm, t
from gp.evalutation_utils import calculate_ci, se_overall_mean_from_cov


def overall_mean(y_pred, y_cov=None, dist=norm, alpha=0.05):
    theta_hat = np.mean(y_pred)
    if y_cov:
        ci = calculate_ci(se_overall_mean_from_cov(y_cov), theta_hat, dist=dist, alpha=alpha)
    return np.mean(y_pred), ci






