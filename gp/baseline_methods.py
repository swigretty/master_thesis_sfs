"""
This module contains all the baseline methods used for predicting BP
values and the function used to generate bootstrapped confidence intervals
associated with them
"""
import numpy as np
from functools import partial
from logging import getLogger
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import KFold

from gp.gp_data import GPData
from gp.gp_plotting_utils import plot_posterior
from target_measures import overall_mean, ttr, TARGET_MEASURES

logger = getLogger(__name__)


def pred_ttr_naive(x_pred, x_train, y_train, **kwargs):
    x_train_unique, unique_idx = np.unique(x_train, return_index=True)
    y_pred = np.repeat(np.nan, len(x_pred))
    y_pred[np.in1d(x_pred, x_train_unique)] = y_train[unique_idx]
    ci_fun = partial(bootstrap, pred_fun=pred_ttr_naive, x_pred=x_pred,
                     x_train=x_train, y_train=y_train)
    return {"data": GPData(x=x_pred, y_mean=y_pred), "ci_fun": ci_fun}


def pred_empirical_mean(x_pred, x_train, y_train, **kwargs):
    """
    Using the overall mean as prediction.
    """
    n = len(y_train)
    y = np.repeat(np.mean(y_train), len(x_pred))

    ci_fun = partial(bootstrap, pred_fun=pred_empirical_mean, x_pred=x_pred,
                     x_train=x_train, y_train=y_train)
    return {"data": GPData(x=x_pred, y_mean=y, y_cov=None), "ci_fun": ci_fun}


def linear_regression(x_pred, x_train, y_train, **kwargs):
    ci = None
    y_cov = None

    t_freq_train = x_train * 2*np.pi * 1/24
    t_freq_pred = x_pred * 2*np.pi * 1/24

    X = np.hstack((np.ones(x_train.shape), x_train, np.sin(t_freq_train),
                   np.cos(t_freq_train)))
    reg = LinearRegression(fit_intercept=False).fit(X, y_train)

    X_pred = np.hstack((np.ones(x_pred.shape), x_pred, np.sin(t_freq_pred),
                        np.cos(t_freq_pred)))
    y = reg.predict(X_pred)

    ci_fun = partial(bootstrap, pred_fun=linear_regression, x_pred=x_pred,
                     x_train=x_train, y_train=y_train)

    return {"data": GPData(x=x_pred, y_mean=y, y_cov=y_cov), "ci_fun": ci_fun}


def mse(true, pred):
    return np.mean((pred - true) ** 2)


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
                  lamds=None, n_knots=None, **kwargs):
    if lamds is None:
        lamds = np.logspace(-1, 2, 10)

    if n_knots is None:
        n_knots = len(np.unique(x_train))
        n_knots = min(n_knots, 100) + int(n_knots**0.2)

    if lamd is None:
        cv_perf = []
        for _lamd in lamds:
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


def loo_cross_val_score(train_x, train_y, fit_pred_fun, cost_function=mse):
    return cross_val_score(train_x, train_y, fit_pred_fun,
                           n_folds=len(train_y), cost_function=cost_function)


def cross_val_score(train_x, train_y, fit_pred_fun, n_folds=10,
                    cost_function=mse):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    scores = []
    for fi, (train_idx, test_idx) in enumerate(kf.split(train_x)):
        pred = fit_pred_fun(train_x[test_idx], train_x[train_idx],
                            train_y[train_idx], train_idx=train_idx,
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


def bootstrap(pred_fun: callable, x_pred: np.ndarray, x_train: np.ndarray,
              y_train: np.ndarray, theta_fun: list[callable] = TARGET_MEASURES,
              n_samples: int = 100, alpha: float = 0.05,
              rng: np.random.Generator = None, output_path: Path = Path(""),
              plot: bool = False, **kwargs) -> dict:
    """
    General function used to create bootstrap confidence intervals
    Returns a dict of the following form:

    ci = {"theta_fun_name": {"mean": theta_hat, "ci_lb": theta_lower_bound,
          "ci_ub": "theta_upper_bound"}


    """

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
    theta_hat = {k: np.apply_along_axis(np.mean, 0, v) for k, v in
                 thetas.items()}
    ci_quant_ub = {k: np.apply_along_axis(partial(np.quantile, q=(alpha/2)),
                                          0, v)
                   for k, v in thetas.items()}
    ci_quant_lb = {k: np.apply_along_axis(partial(np.quantile, q=1-(alpha/2)),
                                          0, v)
                   for k, v in thetas.items()}
    ci = {k: {"mean": theta_hat[k], "ci_lb": 2*theta_hat[k]-ci_quant_lb[k],
          "ci_ub": 2*theta_hat[k]-ci_quant_ub[k]} for k, v in thetas.items()}
    return ci


BASELINE_METHODS = {
    f"naive_{overall_mean.__name__}": pred_empirical_mean,
    f"naive_{ttr.__name__}": pred_ttr_naive,
    "linear": linear_regression,
    "spline": spline_reg_v2,
}


if __name__ == "__main__":
    x = np.linspace(0, 20, 100)
    y = np.sin(x) + np.random.normal() + 120
    train_idx = np.sort(np.random.choice(
        np.arange(len(x)), size=int(len(x) * 0.5), replace=False))
    pred = spline_reg_v2(x_pred=x, x_train=x[train_idx], y_train=y[train_idx])

    fig, ax = plt.subplots()
    ax.plot(pred["data"].x, pred["data"].y_mean)
    ax.plot(x, y)
    ax.scatter(x[train_idx], y[train_idx])
    plt.show()


