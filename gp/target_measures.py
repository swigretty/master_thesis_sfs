import numpy as np
from scipy.stats.distributions import norm, t
from gp.evalutation_utils import calculate_ci, se_overall_mean_from_cov


def ci_overall_mean_gp(y_pred, y_cov=None, alpha=0.05):
    return calculate_ci(se_overall_mean_from_cov(y_cov), overall_mean(y_pred), dist=norm, alpha=alpha)


def overall_mean(y_pred):
    return np.mean(y_pred)


TARGET_MEASURES = [overall_mean]



