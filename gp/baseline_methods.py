import numpy as np
from gp.gp_data import GPData
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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


def linear_regression(x_pred, x_train, y_train):
    t_freq_train = x_train * 2*np.pi * 1/24
    t_freq_pred = x_pred * 2*np.pi * 1/24

    X = np.vstack(np.repeat(1, len(x_train)), x_train, np.sin(t_freq_train), np.cos(t_freq_train))
    reg = LinearRegression(fit_intercept=False).fit(X, y_train)
    residuals = y_train - reg.predict(X)
    RSS = residuals.T @ residuals
    sigma_squared_hat = RSS[0, 0] / (X.shape[0] - X.shape[1])
    XtXinv = np.linalg.inv(X.T @ X)
    # var_beta_hat = XtXinv * sigma_squared_hat
    X_pred = np.vstack(np.repeat(1, len(x_pred)), x_pred, np.sin(t_freq_pred), np.cos(t_freq_pred))
    y = reg.predict(X_pred)

    var_y = sigma_squared_hat * np.sqrt(X_pred.T @ XtXinv @ X_pred)
    y_cov = np.diag(np.repeat(var_y), X_pred.shape[0])
    return GPData(x=x_pred, y_mean=y, y_cov=y_cov)


def linear_regression_statsmodel():
    ols = sm.OLS(y.values, X_with_intercept)
    ols_result = ols.fit()
    print(ols_result.summary())
    return


BASELINE_METHODS = {"naive": pred_empirical_mean}


