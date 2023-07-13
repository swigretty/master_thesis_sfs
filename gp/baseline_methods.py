import numpy as np
from gp.gp_data import GPData


def pred_empirical_mean(x, y):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    sigma_mean = 1 / len(y) * np.var(y)
    y_cov = np.zeros((len(x), len(x)), float)
    np.fill_diagonal(y_cov, sigma_mean)  # iid assumption
    y = np.repeat(np.mean(y), len(x))
    return GPData(x=x, y_mean=y, y_cov=y_cov)