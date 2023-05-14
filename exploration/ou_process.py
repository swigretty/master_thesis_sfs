from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from logging import getLogger
from log_setup import setup_logging
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from exploration.constants import PLOT_PATH
import matplotlib as mpl

logger = getLogger(__name__)
mpl.style.use('seaborn-v0_8')


@dataclass
class OUParams:
    theta: float  # mean reversion parameter
    mu: float  # asymptotic mean
    sigma_w: float  # Brownian motion scale (standard deviation)


@dataclass
class AR1Params:
    const: float  # constant trend/offset (exog in ARIMA output)
    ar_L1: float  # the AR1 coeff
    sigma_w: float  # innovation scale


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
    dW = get_dW(len(t), random_state)
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


def get_cov_ou(ou_params: OUParams, delta_t):
    return ou_params.sigma_w**2/(2*ou_params.theta) * np.exp(-ou_params.theta * delta_t)


if __name__ == "__main__":
    setup_logging()
    # generate process with random_state to reproduce results
    delta_t = 1
    seed = 10
    T = 1000
    t = np.arange(0, T, delta_t)
    OU_params = OUParams(theta=0.1, mu=0.0, sigma_w=0.001)

    delta_t_ar = delta_t #* 0.1
    t_ar = np.arange(0, T, delta_t_ar)
    AR1_params = AR1Params(const=OU_params.theta * OU_params.mu * delta_t_ar, ar_L1=1-OU_params.theta * delta_t_ar,
                           sigma_w=OU_params.sigma_w * np.sqrt(delta_t_ar))

    OU_proc = get_OU_process(T, delta_t, OU_params, random_state=seed)
    OU_params_hat = estimate_OU_params(OU_proc)
    logger.info(f"\n Theoretical {OU_params} \n Estimated {OU_params_hat}")

    arima = ARIMA(OU_proc.astype(np.float), order=(1, 0, 0), trend="c")  # constant trend
    res = arima.fit()  # method="statespace"
    print(res.summary())
    AR1_params_hat = AR1Params(const=res.params[0], ar_L1=res.params[1], sigma_w=np.sqrt(res.params[2]))
    logger.info(f"\n Theoretical {AR1_params} \n Estimated {AR1_params_hat}")

    OU_proc_AR_1 = [0]
    for t_ in t_ar:
        X_t = AR1_params.const + AR1_params.ar_L1 * OU_proc_AR_1[-1] + AR1_params.sigma_w * np.random.standard_normal(1)
        OU_proc_AR_1.append(X_t)
    OU_proc_AR_1 = OU_proc_AR_1[:-1]
        # if t_ in t:
        #     OU_proc_AR_1.append(X_t)



    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t, OU_proc, label="OU")
    ax.plot(t_ar, OU_proc_AR_1, label="AR1")
    ax.legend()
    plt.show()

    ## Theoretical autocovariance
    # AR1
    ar1_cov = arma_acovf(ar=np.array([1, - AR1_params.ar_L1]), ma=np.array([1]), sigma2=AR1_params.sigma_w**2,
                         nobs=len(t_ar))
    cov_0 = AR1_params.sigma_w**2/(1-AR1_params.ar_L1**2)
    cov_1 = cov_0 * AR1_params.ar_L1
    ar1_cov_hat = acovf(OU_proc)

    # COV Matérn Kernel and OU coefficients
    cov_ou = [get_cov_ou(OU_params, delta_t * i) for i in range(len(t))]

    matern_kernel = cov_ou[0] * Matern(nu=0.5, length_scale=1/OU_params.theta)
    cov_matern = matern_kernel(np.array(t).reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.plot(t_ar[: 50 * int(delta_t /delta_t_ar) ], ar1_cov[:50 * int(delta_t /delta_t_ar)], "r.", label="cov_ar1")
    ax.plot(t[:50], cov_matern[0, :50], "b--",  label="matern", )
    ax.plot(t[:50], cov_ou[:50], "y*", label="cov_ou")
    ax.legend()
    fig.savefig(PLOT_PATH / "covariance_ar_ou_matern.pdf")
    plt.show()











