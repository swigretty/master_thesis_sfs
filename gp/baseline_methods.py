import numpy as np
import scipy
from functools import partial
from gp.gp_data import GPData
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from target_measures import overall_mean, ttr, TARGET_MEASURES
from gp.evalutation_utils import calculate_ci
from sklearn.model_selection import LeaveOneOut
from logging import getLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.api as sm
from patsy import dmatrix
from gp.gp_plotting_utils import plot_kernel_function, plot_posterior, plot_gpr_samples, Plotter


logger = getLogger(__name__)


def pred_ttr_naive(x_pred, x_train, y_train, **kwargs):
    x_train_unique, unique_idx = np.unique(x_train, return_index=True)
    y_pred = np.repeat(np.nan, len(x_pred))
    y_pred[np.in1d(x_pred, x_train_unique)] = y_train[unique_idx]
    return {"data": GPData(x=x_pred, y_mean=y_pred)}


def pred_empirical_mean(x_pred, x_train, y_train, return_ci=False):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    n = len(y_train)
    y = np.repeat(np.mean(y_train), len(x_pred))
    cis = {}

    if return_ci:
        ci = bootstrap(pred_empirical_mean, x_pred, x_train, y_train)
        # sem = np.std(y_train) / np.sqrt(len(y_train))
        #
        # cis = {f"ci_{overall_mean.__name__}": calculate_ci(sem, overall_mean(y), dist=scipy.stats.distributions.t(
        #     df=n-1))}

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=None), "ci": ci}


def linear_regression(x_pred, x_train, y_train, return_ci=False):

    y_cov = None

    t_freq_train = x_train * 2*np.pi * 1/24
    t_freq_pred = x_pred * 2*np.pi * 1/24

    X = np.hstack((np.ones(x_train.shape), x_train, np.sin(t_freq_train), np.cos(t_freq_train)))
    reg = LinearRegression(fit_intercept=False).fit(X, y_train)

    X_pred = np.hstack((np.ones(x_pred.shape), x_pred, np.sin(t_freq_pred), np.cos(t_freq_pred)))
    y = reg.predict(X_pred)

    if return_ci:
        ci = bootstrap(linear_regression, x_pred, x_train, y_train)

    # if return_ci:
    #     residuals = y_train - reg.predict(X)
    #     RSS = residuals.T @ residuals
    #     sigma_squared_hat = RSS / (X.shape[0] - X.shape[1])
    #     XtXinv = np.linalg.inv(X.T @ X)
    #     var_y = sigma_squared_hat * np.diag((X_pred @ XtXinv @ X_pred.T))
    #     y_cov = np.diag(var_y)

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=y_cov), "ci": ci}


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
    dfs = np.linspace(3, int(len(x_train)*0.8), 20)
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

    def glm_predict(x_pred, transformed_x, y_train, df):
        cs = sm.GLM(y_train, transformed_x).fit()
        y_pred = cs.predict(
            dmatrix(f"bs(test, degree=3, df={df}, include_intercept=False)", {"test": x_pred},
                    return_type='matrix'))
        return y_pred

    y_pred = glm_predict(x_pred, transformed_x, y_train, df)

    cis = {}
    for tm in TARGET_MEASURES:
        theta_hat, ci = bootstrap(partial(glm_predict, df=df), tm, x_pred, transformed_x, y_train)
        cis[f"ci_{tm.__name__}"] = ci

    # try to get Bspline basis
    # c = np.eye(len(t) - k - 1)
    # k = 3
    # m = len(set(x_train))
    # nest = max(m + k + 1, 2 * k + 3)
    # c = np.empty((nest,), float)
    # from scipy.interpolate import BSpline
    # design_matrix_gh = BSpline(np.unique(x_train), c, k)(x_train)

    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None),
            "fun": partial(spline_reg_v2, df=df), **cis}


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
        # logger.info(f"Best smooting parameter for spline {s}")

    if len(x_train) != m:  # If there are duplicates
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
            "fun": partial(spline_reg, y_std=y_std, lambs=np.linspace(s, s+s*0.3, 3))}


def cross_val_score(train_x, train_y, fit_pred_fun, n_folds=10, cost_function=mse):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    for fi, (train_idx, test_idx) in enumerate(kf.split(train_x)):
        pred = fit_pred_fun(train_x[test_idx], train_x[train_idx], train_y[train_idx])
        scores.append(cost_function(train_y[test_idx], pred["data"].y_mean))
    return np.mean(scores)


def bootstrap(self, pred_fun, pred_x, train_x, train_y, theta_fun=TARGET_MEASURES, n_samples=100, alpha=0.05, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    thetas = {}
    if not isinstance(theta_fun, list):
        theta_fun = theta_fun

    for i in range(n_samples):
        idx = sorted(self.rng.choice(np.arange(len(train_y)), size=len(train_y), replace=True))
        y_sub = train_y[idx]
        x_sub = train_x[idx]
        pred = pred_fun(pred_x, x_sub, y_sub)
        if i == 0:
            fun = pred_fun
            if isinstance(pred_fun, partial):
                fun = pred_fun.func
            fig, ax = plt.subplots()
            plot_posterior(x=pred_x, y_post_mean=pred["data"].y_mean, x_red=x_sub, y_red=y_sub, ax=ax)
            ax.set_title(f"Prediction {fun.__name__}")
            fig.savefig(f"plot_pred_bootstrap_{fun.__name__}.pdf")
        for theta_f in theta_fun:
            theta = theta_f(pred["data"].y_mean)
            thetas_list = thetas.get(theta_f.__name__, [])
            thetas_list.append(theta)

    theta_hat = {k: np.mean(v) for k, v in thetas.items()}
    ci = {k: {"mean": theta_hat[k], "ci_lb": 2*theta_hat[k]-np.quantile(v, 1-(alpha/2)),
          "ci_ub": 2*theta_hat[k]-np.quantile(v, (alpha/2))} for k, v in thetas.items()}
    return ci


BASELINE_METHODS = {f"naive_{overall_mean.__name__}": pred_empirical_mean,
                    f"naive_{ttr.__name__}": pred_ttr_naive,
                    "linear": linear_regression,
                    "spline": spline_reg,
                    "splinev2": spline_reg_v2}


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

