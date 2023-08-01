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
import statsmodels.api as sm
from patsy import dmatrix


logger = getLogger(__name__)


def pred_empirical_mean(x_pred, x_train, y_train, return_ci=False):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
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


def mse(true, pred):
    return np.mean((pred - true) ** 2)


def get_rep_count_cluster(x_train):
    """
    returns {start_idx_cluster: number_of_duplicates}
    """
    diffs = np.diff(x_train)
    rep_count = {}
    current_cluster = 0
    rep_count[current_cluster] = 1
    for i, diff in enumerate(diffs):
        if diff == 0:
            rep_count[current_cluster] += 1
        else:
            current_cluster = i+1
            rep_count[current_cluster] = 1
    return rep_count


def spline_reg_v2(x_pred, x_train, y_train, df=None, **kwargs):
    dfs = np.array([3, 6, 12, 24, 50, 100])
    assert all(sorted(x_train) == x_train)

    if df is None:
        cv_perf = []
        for _df in dfs:
            fit_pred_fun = partial(spline_reg_v2, df=_df)
            cv_perf.append(cross_val_score(train_x=x_train, train_y=y_train, fit_pred_fun=fit_pred_fun))

        df = dfs[np.where(cv_perf == np.min(cv_perf))][0]
        logger.info(f"Best smoothing parameter for spline {df=}")

    transformed_x = dmatrix(
        f"bs(train, degree=3, df={df}, include_intercept=False)",
        {"train": x_train}, return_type='dataframe')
    cs = sm.GLM(y_train, transformed_x).fit()
    y_pred = cs.predict(
        dmatrix(f"bs(test, degree=3, df={df}, include_intercept=False)", {"test": x_pred},
                return_type='matrix'))

    # try to get Bspline basis
    # c = np.eye(len(t) - k - 1)
    # k = 3
    # m = len(set(x_train))
    # nest = max(m + k + 1, 2 * k + 3)
    # c = np.empty((nest,), float)
    # from scipy.interpolate import BSpline
    # design_matrix_gh = BSpline(np.unique(x_train), c, k)(x_train)
    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None),
            "fun": partial(spline_reg_v2, df=df)}


def spline_reg(x_pred, x_train, y_train, s=None, y_std=1, normalize_y=True, lambs=None, **kwargs):
    m = len(np.unique(x_train))

    if lambs is None:
        lambs = np.linspace(max(m - np.sqrt(2 * m) - m * 0.8, 10), m + np.sqrt(2 * m) + m * 0.3, 10)

    x_train = x_train.reshape(-1)
    assert all(sorted(x_train) == x_train)
    y_train = y_train.reshape(-1)
    x_pred = x_pred.reshape(-1)

    if normalize_y:
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        y_train = (y_train-y_train_mean)/y_train_std

    weights = np.repeat(1/y_std, len(x_train))

    if s is None:
        cv_perf = []
        for lamb in lambs:
            fit_pred_fun = partial(spline_reg, y_std=y_std, s=lamb, normalize_y=False)
            cv_perf.append(cross_val_score(train_x=x_train, train_y=y_train, fit_pred_fun=fit_pred_fun))
        idx_numeric = np.where(~ np.isnan(cv_perf))[0]
        s = lambs[np.where(cv_perf == np.min(np.array(cv_perf)[idx_numeric]))][0]
        logger.info(f"Best smooting parameter for spline {s}")

    if len(x_train) != m:  # If Tthere are duplicates
        rep_count = get_rep_count_cluster(x_train)
        # Weight is number of repetitions of same datapoint (bootstrap)
        weights = np.array(list(rep_count.values()))
        unique_idx = np.array(list(rep_count.keys()))
        x_train = x_train[unique_idx]
        y_train = y_train[unique_idx]

    # Strictly positive rank-1 array of weights the same length as x and y.
    # The weights are used in computing the weighted least-squares spline fit.
    # If the errors in the y values have standard-deviation given by the vector d,
    # then w should be 1/d. Default is ones(len(x)).
    spl = scipy.interpolate.UnivariateSpline(x_train, y_train, s=s, w=weights, ext=1)
    # ext=1, extrapolate with 0, ext=3: extrapolate with boundary value
    y_pred = spl(x_pred)

    if normalize_y:
        y_pred = y_pred * y_train_std + y_train_mean

    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None),
            "fun": partial(spline_reg, y_std=y_std, lambs=np.linspace(s-s*0.5, s+s*0.5, 5))}


def cross_val_score(train_x, train_y, fit_pred_fun, n_folds=10, cost_function=mse):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    for fi, (train_idx, test_idx) in enumerate(kf.split(train_x)):
        pred = fit_pred_fun(train_x[test_idx], train_x[train_idx], train_y[train_idx])
        scores.append(cost_function(train_y[test_idx], pred["data"].y_mean))
    return np.mean(scores)


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

