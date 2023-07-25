import numpy as np
import scipy
from functools import partial
from gp.gp_data import GPData
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from target_measures import overall_mean
from gp.evalutation_utils import calculate_ci
from sklearn.model_selection import LeaveOneOut
from logging import getLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


logger = getLogger(__name__)

def pred_empirical_mean(x_pred, x_train, y_train, return_ci=False):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    logger.info("Extracting empirical Mean")
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
    logger.info("Fitting linear regression")

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


def cross_val_score(self, train_x, train_y, n_folds=3):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    for fi, (train_idx, test_idx) in enumerate(kf.split(train_x)):
        self.gp.fit(train_x[train_idx], train_y[train_idx])
        predictions = self.gp.predict(train_x[test_idx])
        scores.append(cost_function(train_y[test_idx], predictions))
    return np.mean(scores)

def spline_reg(x_pred, x_train, y_train, **kwargs):
    logger.info("Fitting spline")

    lambs = np.array([0.8])
    mse = []
    loo = LeaveOneOut()
    x_train = x_train.reshape(-1)
    assert all(sorted(x_train) == x_train)
    y_train = y_train.reshape(-1)
    x_pred = x_pred.reshape(-1)

    # for i in lambs:
    #     error = []
    #     for trg, tst in loo.split(x_train):
    #         try:
    #             spl = scipy.interpolate.splrep(x_train[trg], y_train[trg], s=i, per=True)
    #         except ValueError as e:
    #             logger.warning(f"Could not fit spline: {e}")
    #             break
    #         pred = scipy.interpolate.splev(x_train[tst], spl, ext=1)[0]
    #         if np.isnan(pred):
    #             break
    #         true = y_train[tst][0]
    #         error.append((pred - true) ** 2)
    #
    #     mse.append(np.mean(error))
    # lamb_best = lambs[np.where(mse == np.min(mse))][0]
    y_std = np.std(y_train)
    weights = np.repeat(1/y_std, len(x_train))

    lamb_best = len(x_train)-np.sqrt(2*len(x_train))
    lamb_best = 10
    x_train_rep = [i for i in range(1, len(x_train)) if x_train[i] == x_train[i-1]]
    if x_train_rep:
        unique_idx = np.setdiff1d(range(len(x_train)), x_train_rep)
        weights[np.array(x_train_rep)-1] = 2/y_std
        x_train = x_train[unique_idx]
        y_train = y_train[unique_idx]
        weights = weights[unique_idx]

    spl = scipy.interpolate.splrep(x_train, y_train, w=weights, per=True, s=lamb_best)
    y_pred = scipy.interpolate.splev(x_pred, spl, ext=3)

    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None)}


BASELINE_METHODS = {"naive": pred_empirical_mean, "linear": linear_regression, "spline": spline_reg}


if __name__ == "__main__":

    # x = np.arange(100)
    # y = np.sin(x)
    x = np.linspace(0, 100, 200)
    y = np.sin(x)
    train_idx = np.sort(np.random.choice(np.arange(len(x)), size=int(len(x) * 0.8), replace=False))
    # test_idx = np.setdiff1d(x, train_idx)

    pred = spline_reg(x_pred=x, x_train=x[train_idx], y_train=y[train_idx])

    fig, ax = plt.subplots()
    ax.plot(pred["data"].x, pred["data"].y_mean)
    plt.show()

    print()

