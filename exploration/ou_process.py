from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from logging import getLogger
from log_setup import setup_logging
from statsmodels.tsa.arima_process import arma_acovf

import matplotlib.pyplot as plt

logger = getLogger(__name__)

@dataclass
class OUParams:
    theta: float  # mean reversion parameter
    mu: float  # asymptotic mean
    sigma_w: float  # Brownian motion scale (standard deviation)


def get_OU_process(
    T: int,
    delta_t,
    OU_params: OUParams,
    X_0: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    - T is the sample size.
    - Ou_params is an instance of OUParams dataclass.
    - X_0 the initial value for the process, if None, then X_0 is taken
        to be mu (the asymptotic mean).
    Returns a 1D array.
    """
    t = np.arange(0, T, delta_t, dtype=np.float128)  # float to avoid np.exp overflow
    exp_theta_t = np.exp(-OU_params.theta * t)
    dW = get_dW(T, random_state)
    integral_W = _get_integal_W(t, dW, OU_params)
    _X_0 = _select_X_0(X_0, OU_params)
    return (
        _X_0 * exp_theta_t
        + OU_params.mu * (1 - exp_theta_t)
        + OU_params.sigma_w * exp_theta_t * integral_W
    )


def _select_X_0(X_0_in: Optional[float], OU_params: OUParams) -> float:
    """Returns X_0 input if not none, else mu (the long term mean)."""
    if X_0_in is not None:
        return X_0_in
    return OU_params.mu


def _get_integal_W(
    t: np.ndarray, dW: np.ndarray, OU_params: OUParams
) -> np.ndarray:
    """Integral with respect to Brownian Motion (W), ∫...dW."""
    exp_theta_s = np.exp(OU_params.theta * t)
    integral_W = np.cumsum(exp_theta_s * dW)
    return np.insert(integral_W, 0, 0)[:-1]


def get_dW(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample T times from a normal distribution,
    to simulate discrete increments (dW) of a Brownian Motion.
    Optional random_state to reproduce results.
    """
    np.random.seed(random_state)
    return np.random.normal(0.0, 1.0, T)


def estimate_OU_params(X_t: np.ndarray) -> OUParams:
    """
    Estimate OU params from OLS regression.
    - X_t is a 1D array.
    Returns instance of OUParams.
    """
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    theta = -reg.coef_[0]
    mu = reg.intercept_ / theta
    # residuals and their standard deviation
    y_hat = reg.predict(X)
    sigma_w = np.std(y - y_hat)
    return OUParams(theta, mu, sigma_w)


if __name__ == "__main__":
    setup_logging()
    # generate process with random_state to reproduce results
    delta_t = 1
    seed = 10
    T = 100_000
    t = np.arange(0, T, delta_t)

    OU_params = OUParams(theta=0.07, mu=0.0, sigma_w=0.001)
    OU_proc = get_OU_process(T, delta_t, OU_params, random_state=seed)

    OU_params_hat = estimate_OU_params(OU_proc)
    logger.info(OU_params_hat)
    logger.info("bla")

    c = OU_params.theta * OU_params.mu * delta_t
    a = - OU_params.theta * delta_t + 1
    b = OU_params.sigma_w * np.sqrt(delta_t)

    innovations = b * get_dW(T, random_state=seed)

    OU_proc_AR_1 = [0]
    for w in innovations:
        OU_proc_AR_1.append(c + a * OU_proc_AR_1[-1] + w)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t, OU_proc, label="OU")
    ax.plot(t, OU_proc_AR_1[:-1], label="AR1")
    ax.legend()
    plt.show()


    # Estimated autocovariance


    # Theoretical autocovariance

    # COV AR(1)
    cov_true = arma_acovf(ar=a, ma=np.array([1]), sigma2=b, nobs=len(t))
    cov_0 = b**2/(1-a**2)
    cov_1 = cov_0 * a**2




