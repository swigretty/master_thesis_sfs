import numpy as np
from scipy.stats.distributions import norm, t
from gp.evalutation_utils import calculate_ci, se_overall_mean_from_cov


def ci_overall_mean_gp(y_pred, y_cov=None, alpha=0.05):
    return calculate_ci(se_overall_mean_from_cov(y_cov), overall_mean(y_pred), dist=norm, alpha=alpha)


def overall_mean(y_pred, x_pred=None):
    return np.mean(y_pred)


def get_cycles(x_pred, cycle_length):
    return (x_pred/cycle_length).astype(int).reshape(-1)


def get_cycle_length_daily(x_unit):
    if x_unit == "hour":
        cycle_length = 24
    return cycle_length


def cis_mean_24h_gp(x_pred, y_pred, y_cov=None, alpha=0.05, x_unit="hour"):
    cycles = get_cycles(x_pred, get_cycle_length_daily(x_unit))
    mean_ci_cylces = {}

    for cn in range(np.max(cycles)):
        idx_cycle = cycles == cn
        mean = np.mean(y_pred[idx_cycle])
        cov = y_cov[idx_cycle, idx_cycle]
        ci = calculate_ci(se_overall_mean_from_cov(cov), mean, dist=norm, alpha=alpha)
        mean_ci_cylces[cn] = {"mean": mean, "ci": ci}
    return cis_mean_24h_gp


def mean_cycle(y_pred, x_pred, cycle_length):
    cycles = get_cycles(x_pred, cycle_length)
    # {cycle_number: mean_value_of_cycle}
    # If there is not any not nan value in the cylce put np.nan
    mean_cycles = {cn: (np.nanmean(y_pred[cycles == cn]) if np.any(~np.isnan(y_pred[cycles == cn])) else np.nan) for cn
                   in range(np.max(cycles))}
    out_array = np.array(list(mean_cycles.values()))
    # Inpute nans with mean
    out_array[np.isnan(out_array)] = np.nanmean(y_pred)
    return out_array


def mean_24h(y_pred, x_pred, x_unit="hour"):
    cycle_legth = get_cycle_length_daily(x_unit)
    return mean_cycle(y_pred, x_pred, cycle_legth)


def mean_1h(y_pred, x_pred,  x_unit="hour"):
    return mean_cycle(y_pred, x_pred, 1)


def raw(y_pred, x_pred=None):
    return y_pred


def ttr(y_pred, x_pred=None, thr_lower=90-120, thr_upper=125-120):
    """
    24h: 90 to 125 (for systolic BP)
    """
    y_pred_num = y_pred[~np.isnan(y_pred)]
    n_in_range = sum((thr_lower <= y_pred_num) & (y_pred_num <= thr_upper))
    n_total = len(y_pred_num)
    return n_in_range/n_total


TARGET_MEASURES_LIST = [overall_mean, ttr, mean_1h, mean_24h, raw]

TARGET_MEASURES = {f.__name__: f for f in TARGET_MEASURES_LIST}

