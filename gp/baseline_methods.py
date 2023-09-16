import numpy as np
import scipy
from functools import partial
from gp.gp_data import GPData
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
from target_measures import overall_mean, ttr, TARGET_MEASURES
from gp.evalutation_utils import calculate_ci
from sklearn.model_selection import LeaveOneOut
from logging import getLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.api as sm
import patsy
from scipy import stats
from pathlib import Path
from gp.gp_plotting_utils import plot_kernel_function, plot_posterior, plot_gpr_samples, Plotter
from sklearn.preprocessing import SplineTransformer
from pygam import GAM, s, te, l, LinearGAM


logger = getLogger(__name__)


def conf_int_linear_regression(x_train, y_train, model, x_pred, alpha=0.05):
    # degrees of freedom
    dof = -np.diff(x.shape)[0]
    # Student's t-distribution table lookup
    t_val = stats.t.isf(alpha / 2, dof)
    residuals = y_train - model.predict(x_train)
    rss = residuals.T @ residuals
    sigma_squared_hat = rss / (x_train.shape[0] - x_train.shape[1])
    xtxinv = np.linalg.inv(x_train.T @ x_train)
    var_y_pred = sigma_squared_hat * np.diag((x_pred @ xtxinv @ x_pred.T))

    # # rss = np.sum((Y_train - lin_model.predict(X_train)) ** 2) / dof
    # # inverse of the variance of the parameters
    # var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    # # distance between lower and upper bound of CI
    # gap = t_val * np.sqrt(rss * var_params)
    #
    # ci = {k: {"mean": theta_hat[k], "ci_lb": 2*theta_hat[k]-np.quantile(v, 1-(alpha/2)),
    #       "ci_ub": 2*theta_hat[k]-np.quantile(v, (alpha/2))} for k, v in thetas.items()}
    #
    # return ci


def pred_ttr_naive(x_pred, x_train, y_train, **kwargs):
    ci = None

    x_train_unique, unique_idx = np.unique(x_train, return_index=True)
    y_pred = np.repeat(np.nan, len(x_pred))
    y_pred[np.in1d(x_pred, x_train_unique)] = y_train[unique_idx]
    ci_fun = partial(bootstrap, pred_fun=pred_ttr_naive, x_pred=x_pred, x_train=x_train, y_train=y_train)
    return {"data": GPData(x=x_pred, y_mean=y_pred), "ci_fun": ci_fun}


def pred_empirical_mean(x_pred, x_train, y_train, **kwargs):
    """
    Using the overall mean as prediction.
    cov is calculated assuming iid data.
    """
    n = len(y_train)
    y = np.repeat(np.mean(y_train), len(x_pred))

    ci_fun = partial(bootstrap, pred_fun=pred_empirical_mean, x_pred=x_pred, x_train=x_train, y_train=y_train)
        # sem = np.std(y_train) / np.sqrt(len(y_train))
        # cis = {f"ci_{overall_mean.__name__}": calculate_ci(sem, overall_mean(y), dist=scipy.stats.distributions.t(
        #     df=n-1))}

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=None), "ci_fun": ci_fun}


def linear_regression(x_pred, x_train, y_train, **kwargs):
    ci = None
    y_cov = None

    t_freq_train = x_train * 2*np.pi * 1/24
    t_freq_pred = x_pred * 2*np.pi * 1/24

    X = np.hstack((np.ones(x_train.shape), x_train, np.sin(t_freq_train), np.cos(t_freq_train)))
    reg = LinearRegression(fit_intercept=False).fit(X, y_train)

    X_pred = np.hstack((np.ones(x_pred.shape), x_pred, np.sin(t_freq_pred), np.cos(t_freq_pred)))
    y = reg.predict(X_pred)

    ci_fun = partial(bootstrap, pred_fun=linear_regression, x_pred=x_pred, x_train=x_train, y_train=y_train)


    return {"data": GPData(x=x_pred, y_mean=y, y_cov=y_cov), "ci_fun": ci_fun}


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


def get_spline_basis(x_pred, x_train, n_knots=None):

    if n_knots is None:
        n_knots = len(np.unique(x_train))
    spline = SplineTransformer(degree=3, n_knots=n_knots,
                               extrapolation="constant", knots="quantile")  #

    spline.fit(x_train)

    x_train_trans = spline.transform(x_train)
    x_pred_trans = spline.transform(x_pred)

    return x_pred_trans, x_train_trans


def spline_reg_v2(x_pred, x_train, y_train, lamd=None, transformed=False,
                  lamds=None, n_knots=None, train_idx=None, test_idx=None,
                  **kwargs):
    if lamds is None:
        lamds = np.logspace(-1, 2, 10)

    if n_knots is None:
        n_knots = len(np.unique(x_train))
        n_knots = min(n_knots, 100) + int(n_knots**0.2)

    if lamd is None:
        cv_perf = []
        for _lamd in lamds:
            # _, x_train_trans = get_spline_basis(x_pred, x_train, _df)
            fit_pred_fun = partial(spline_reg_v2, lamd=_lamd, n_knots=n_knots)
            cv_perf.append(cross_val_score(train_x=x_train, train_y=y_train,
                                           fit_pred_fun=fit_pred_fun,
                                           n_folds=10))

        lamd = lamds[np.where(cv_perf == np.min(cv_perf))][0]
        logger.info(f"Best smoothing parameter for spline {lamd=}")

    if transformed:
        x_train_trans = x_train
        x_pred_trans = x_pred
    else:
        x_pred_trans, x_train_trans = get_spline_basis(x_pred, x_train,
                                                       n_knots=n_knots)

    glm = Ridge(alpha=lamd).fit(x_train_trans, y_train)
    y_pred = glm.predict(x_pred_trans)
    ci_fun = partial(bootstrap, pred_fun=partial(spline_reg_v2, lamd=lamd),
                     x_pred=x_pred, x_train=x_train, y_train=y_train)

    if transformed:
        x_pred = None
    else:
        x_pred = x_pred.reshape(-1, 1)

    return {"data": GPData(x=x_pred, y_mean=y_pred, y_cov=None),
            "ci_fun": ci_fun}


def smoothing_spline_prediction(X_train: np.ndarray, Y_X_train: np.ndarray,
                                X: np.ndarray,
                                n_knots: int) -> np.ndarray:
    """
    Parameters
    ----------
    X_train: The training time indexes
    Y_X_train: The training BP values
    X: The time indexes at which to generate predictions
    n_knots : Number of knots of the splines

    Returns
    ---------
    F_X_hat: The BP value predictions at inputs X
    """

    spline = SplineTransformer(degree=3, n_knots=n_knots,
                               extrapolation="constant",
                               knots="quantile")

    spline.fit(X_train)

    X_train_trans = spline.transform(X_train)
    X_trans = spline.transform(X)

    lm = LinearRegression(fit_intercept=False).fit(X_train_trans, Y_X_train)
    F_X_hat = lm.predict(X_trans)
    return F_X_hat


def gam_spline(x_pred, x_train, y_train, normalize_y=True, **kwargs):

    if normalize_y:
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        y_train = (y_train-y_train_mean)/y_train_std

    gam = LinearGAM(s(0))
    gam.fit(x_train, y_train)
    y_pred = gam.predict(x_pred)

    ci_fun = partial(bootstrap, pred_fun=gam_spline,
                     x_pred=x_pred, x_train=x_train, y_train=y_train)

    if normalize_y:
        y_pred = y_pred * y_train_std + y_train_mean

    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None), "ci_fun": ci_fun}


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

    ci_fun = partial(bootstrap, pred_fun=partial(spline_reg, y_std=y_std, lambs=np.linspace(s, s+s*0.3, 3)),
                     x_pred=x_pred, x_train=x_train, y_train=y_train)

    return {"data": GPData(x=x_pred.reshape(-1, 1), y_mean=y_pred, y_cov=None), "ci_fun": ci_fun,
            "fun": partial(spline_reg, y_std=y_std, s=s)}


def loo_cross_val_score(train_x, train_y, fit_pred_fun, cost_function=mse):
    return cross_val_score(train_x, train_y, fit_pred_fun, n_folds=len(train_y), cost_function=cost_function)


def cross_val_score(train_x, train_y, fit_pred_fun, n_folds=10, cost_function=mse):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    # if train_x.ndim == 1:
    #     train_x = train_x.reshape(-1, 1)
    for fi, (train_idx, test_idx) in enumerate(kf.split(train_x)):
        pred = fit_pred_fun(train_x[test_idx], train_x[train_idx], train_y[train_idx], train_idx=train_idx,
                            test_idx=test_idx)
        scores.append(cost_function(train_y[test_idx], pred["data"].y_mean))
    return np.mean(scores)


def plot_bootstrap(pred_fun, x_pred, pred, x_sub, y_sub, output_path, i):
    fun = pred_fun
    if isinstance(pred_fun, partial):
        fun = pred_fun.func
    fig, ax = plt.subplots()
    try:
        plot_posterior(x=x_pred, y_post_mean=pred["data"].y_mean, x_red=x_sub,
                       y_red=y_sub, ax=ax)
        fig.savefig(
            output_path / f"plot_pred_bootstrap_{fun.__name__}_{i}.pdf")
    except Exception as e:
        logger.info(f"could not plot bootstrap sample {e}")
    plt.close(fig=fig)


def bootstrap(pred_fun, x_pred, x_train, y_train, theta_fun=TARGET_MEASURES,
              n_samples=100, alpha=0.05,
              rng=None, logger=logger, output_path=Path("."), plot=False):

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(theta_fun, dict):
        theta_fun = {theta_fun.__name__: theta_fun}

    thetas = {fn: [] for fn, fun in theta_fun.items()}

    for i in range(n_samples):
        idx = sorted(rng.choice(np.arange(len(y_train)), size=len(y_train),
                                replace=True))
        y_sub = y_train[idx]
        x_sub = x_train[idx, ]
        pred = pred_fun(x_pred, x_sub, y_sub, train_idx=idx)
        if plot and i % int(n_samples/10) == 0:
            plot_bootstrap(pred_fun, x_pred, pred, x_sub, y_sub,
                           output_path, i)
        for fn, theta_f in theta_fun.items():
            thetas[fn].append(theta_f(pred["data"].y_mean))

    thetas = {k: np.array(v) for k, v in thetas.items()}
    theta_hat = {k: np.apply_along_axis(np.mean, 0, v) for k, v in thetas.items()}
    ci_quant_ub = {k: np.apply_along_axis(partial(np.quantile, q=(alpha/2)), 0, v)
                   for k, v in thetas.items()}
    ci_quant_lb = {k: np.apply_along_axis(partial(np.quantile, q=1-(alpha/2)), 0, v)
                   for k, v in thetas.items()}
    ci = {k: {"mean": theta_hat[k], "ci_lb": 2*theta_hat[k]-ci_quant_lb[k],
          "ci_ub": 2*theta_hat[k]-ci_quant_ub[k]} for k, v in thetas.items()}
    return ci


BASELINE_METHODS = {
    f"naive_{overall_mean.__name__}": pred_empirical_mean,
    f"naive_{ttr.__name__}": pred_ttr_naive,
    "linear": linear_regression,
    "spline": spline_reg_v2,
    # "gam_spline": gam_spline
}


if __name__ == "__main__":

    # x = np.arange(100)
    # y = np.sin(x)
    x = np.linspace(0, 20, 100)
    y = np.sin(x) + np.random.normal() + 120
    train_idx = np.sort(np.random.choice(np.arange(len(x)), size=int(len(x) * 0.5), replace=False))
    # test_idx = np.setdiff1d(x, train_idx)

    # pred = spline_reg(x_pred=x, x_train=x[train_idx], y_train=y[train_idx])
    pred = spline_reg_v2(x_pred=x, x_train=x[train_idx], y_train=y[train_idx])

    fig, ax = plt.subplots()
    ax.plot(pred["data"].x, pred["data"].y_mean)
    ax.plot(x, y)
    ax.scatter(x[train_idx], y[train_idx])
    plt.show()

    print()

