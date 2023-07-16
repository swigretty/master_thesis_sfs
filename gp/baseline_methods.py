import numpy as np
from gp.gp_data import GPData


def pred_empirical_mean(x_pred, x_train, y_train):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    sigma_mean = 1 / len(y_train) * np.var(y_train)
    y_cov = np.zeros((len(x_pred), len(x_pred)), float)
    np.fill_diagonal(y_cov, sigma_mean)  # iid assumption
    y = np.repeat(np.mean(y_train), len(x_pred))
    return GPData(x=x_pred, y_mean=y, y_cov=y_cov)


