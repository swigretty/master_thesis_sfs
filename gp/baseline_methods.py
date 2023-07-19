import numpy as np
import scipy
from functools import partial
from gp.gp_data import GPData
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from target_measures import overall_mean
from gp.evalutation_utils import calculate_ci


def pred_empirical_mean(x_pred, x_train, y_train, return_ci=False):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    y_train_mean = np.mean(y_train)
    n = len(y_train)
    y = np.repeat(np.mean(y_train), len(x_pred))
    cis = {}

    if return_ci:
        sem = np.std(y_train) / np.sqrt(len(y_train))

        cis = {f"ci_{overall_mean.__name__}": calculate_ci(sem, overall_mean(y), dist=scipy.stats.distributions.t(
            df=n-1))}

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=None), **cis}


def linear_regression(x_pred, x_train, y_train, return_ci=False):
    y_cov = None

    t_freq_train = x_train * 2*np.pi * 1/24
    t_freq_pred = x_pred * 2*np.pi * 1/24

    X = np.hstack((np.ones(x_train.shape), x_train, np.sin(t_freq_train), np.cos(t_freq_train)))
    reg = LinearRegression(fit_intercept=False).fit(X, y_train)

    X_pred = np.hstack((np.ones(x_pred.shape), x_pred, np.sin(t_freq_pred), np.cos(t_freq_pred)))
    y = reg.predict(X_pred)

    # if return_ci:
    #     residuals = y_train - reg.predict(X)
    #     RSS = residuals.T @ residuals
    #     sigma_squared_hat = RSS / (X.shape[0] - X.shape[1])
    #     XtXinv = np.linalg.inv(X.T @ X)
    #     var_y = sigma_squared_hat * np.diag((X_pred @ XtXinv @ X_pred.T))
    #     y_cov = np.diag(var_y)

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=y_cov)}


def linear_regression_statsmodel():
    ols = sm.OLS(y.values, X_with_intercept)
    ols_result = ols.fit()
    print(ols_result.summary())
    return


BASELINE_METHODS = {"naive": pred_empirical_mean, "linear": linear_regression}


