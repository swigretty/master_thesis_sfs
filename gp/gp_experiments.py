import copy

from gp.simulate_gp import GPSimulator
from gp.simulate_gp_config import OU_KERNELS, base_config
from gp.post_sim_analysis import perf_plot, perf_plot_split
from logging import getLogger
from log_setup import setup_logging
import numpy as np
import pandas as pd
from constants.constants import get_output_path

from datetime import datetime


logger = getLogger(__name__)

MODES = {
    "ou_bounded": {"kernels": OU_KERNELS["bounded"], "config": {"normalize_y": False, **base_config}},
    "ou_bounded_seasonal": {"kernels": OU_KERNELS["bounded"],
                            "config": {"normalize_y": False, "data_fraction_weights": "seasonal",
                                       **base_config}}}


def plot_gp_regression_sample(session_name=None, nplots=1, rng=None, normalize_kernel=False, **kwargs):
    mode_name_suffix = ""
    if rng is None:
        rng = np.random.default_rng(11)
    for i in range(nplots):
        if nplots > 1:
            rng = np.random.default_rng(i)
            mode_name_suffix = f"_{i}"
        if session_name is not None:
            session_name = f"{session_name}{mode_name_suffix}"
        gps = GPSimulator(rng=rng, normalize_kernel=normalize_kernel, session_name=session_name, **kwargs)
        gps.plot_true_with_samples()
        gps.plot()
        gps.plot_errors()
        gps.plot_overall_mean()

    return gps


def evaluate_multisample(gps=None, n_samples=10, **gps_kwargs):
    if gps is None:
        gps = GPSimulator(normalize_kernel=True, **gps_kwargs)
    eval_dict = gps.evaluate_multisample(n_samples=n_samples)
    return eval_dict


def plot_evaluate_multisample(session_name=None, nplots=1, n_samples=100, normalize_kernel=False, **gps_kwargs):
    # return evaluate_multisample(n_samples=n_samples, session_name=session_name, **gps_kwargs)
    gps = plot_gp_regression_sample(session_name=session_name, nplots=nplots, normalize_kernel=normalize_kernel,
                                    **gps_kwargs)
    return evaluate_multisample(gps=gps, n_samples=n_samples), gps


def evaluate_data_fraction(modes, data_fraction=(0.1, 0.2, 0.4), meas_noise_var=(0.1, 1, 10), n_samples=100):
    output_path = get_output_path()

    eval_row = 0
    for nv in meas_noise_var:
        for mode_name, mode_config in modes.items():
            for k_name, k in mode_config["kernels"].items():
                session_name = f"{mode_name}_{k_name}"
                kernel, nv = GPSimulator.get_normalized_kernel(kernel=k, meas_noise_var=nv)
                for df in data_fraction:
                    logger.info(f"Simulation started for {session_name=}, {df=} and {nv=} ")
                    rng = np.random.default_rng(11)
                    eval_dict, gps = plot_evaluate_multisample(
                        session_name=session_name, rng=rng, kernel_sim=kernel, data_fraction=df, n_samples=n_samples,
                        normalize_kernel=False, meas_noise_var=nv, **mode_config["config"])
                    for k, v in eval_dict.items():
                        df = pd.DataFrame([v])
                        df["mode"] = mode_name
                        df["kernel_name"] = k_name
                        if not (output_path / f"{k}.csv").exists() and eval_row == 0:
                            df.to_csv(output_path / f"{k}.csv", index=None)
                        else:
                            df.to_csv(output_path / f"{k}.csv", mode='a', header=False, index=None)

                    eval_row += 1

    for split in ["overall", "train", "test"]:
        perf_plot(split=split, mode=mode_name, file_path=output_path)


def get_limited_modes(kernels_limited=None, modes_limited=None):
    modes = copy.deepcopy(MODES)
    if modes_limited:
        modes = {k: v for k, v in modes.items() if k in modes_limited}
    if kernels_limited:
        for m_k, m_v in modes.items():
            m_v["kernels"] = {k_k: k_v for k_k, k_v in m_v["kernels"].items() if k_k in kernels_limited}
    return modes


def plot_sample(k_name="sin_rbf", mode_name="ou_bounded_seasonal", data_fraction=0.2, meas_noise_var=1,
                normalize_kernel=True):
    mode = MODES[mode_name]
    session_name = f"{mode_name}_{k_name}"
    return plot_gp_regression_sample(session_name=session_name, kernel_sim=mode["kernels"][k_name],
                                     data_fraction=data_fraction, meas_noise_var=meas_noise_var, **mode["config"],
                                     normalize_kernel=normalize_kernel)


if __name__ == "__main__":
    setup_logging()

    kernels_limited = ["sin_rbf"]
    modes_limited = ["ou_bounded"]
    modes = get_limited_modes(kernels_limited=kernels_limited, modes_limited=modes_limited)

    # plot_sample()
    evaluate_data_fraction(modes, meas_noise_var=(0.1, 1), data_fraction=(0.2,), n_samples=1)

    for split in ["overall", "train", "test"]:
        perf_plot(split=split, file_path=get_output_path())
