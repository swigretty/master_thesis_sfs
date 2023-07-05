from gp.simulate_gp import GPSimulator
from gp.simulate_gp_config import OU_KERNELS, base_config
from gp.post_sim_analysis import perf_plot, perf_plot_split
from logging import getLogger
from log_setup import setup_logging
from constants.constants import OUTPUT_PATH
import numpy as np
import pandas as pd
from datetime import datetime


logger = getLogger(__name__)


def plot_gp_regression_sample(k_name, mode_name, nplots=1, rng=None, **kwargs):
    mode_name_suffix = ""
    if rng is None:
        rng = np.random.default_rng(11)
    for i in range(nplots):
        if nplots > 1:
            rng = np.random.default_rng(i)
            mode_name_suffix = f"_{i}"

        gps = GPSimulator(rng=rng, normalize_kernel=True, session_name=f"{mode_name}_{k_name}{mode_name_suffix}",
                          **kwargs)
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


def plot_evaluate_multisample(k_name, mode_name, nplots=1, n_samples=100, **gps_kwargs):
    gps = plot_gp_regression_sample(k_name, mode_name, nplots=nplots, **gps_kwargs)
    return evaluate_multisample(gps=gps, n_samples=n_samples)


def plot_evaluate_kernels(modes, data_fraction_list, rng=np.random.default_rng(11), n_samples=100):
    eval_row = 0

    for mode_name, mode_config in modes.items():
        for data_fraction in data_fraction_list:
            for k_name, k in mode_config["kernels"].items():
                logger.info(f"Simulation started for {mode_name}: {k_name}")
                eval_dict = plot_evaluate_multisample(k_name, mode_name, rng=rng, kernel_sim=k,
                                                      data_fraction=data_fraction, **mode_config["config"],
                                                      n_samples=n_samples)

                for k, v in eval_dict.items():
                    df = pd.DataFrame([v])
                    df["mode"] = mode_name
                    df["kernel_name"] = k_name
                    if not (OUTPUT_PATH / f"{k}.csv").exists() and eval_row == 0:
                        df.to_csv(OUTPUT_PATH / f"{k}.csv")
                    else:
                        df.to_csv(OUTPUT_PATH / f"{k}.csv", mode='a', header=False)

                eval_row += 1

        for split in ["overall", "train", "test"]:
            perf_plot(split=split, mode=mode_name)


if __name__ == "__main__":

    setup_logging()
    nplots = 3
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    data_fraction_list = [0.1, 0.2, 0.4]

    kernels_limited = None
    kernels_limited = ["sin_rbf"]

    modes = {
        "ou_bounded": {"kernels":  OU_KERNELS["bounded"], "config": {"normalize_y": False, **base_config}},
        "ou_bounded_seasonal": {"kernels": OU_KERNELS["bounded"],
                                "config": {"normalize_y": False, "data_fraction_weights": "seasonal",
                                           **base_config}}}
    for m_k, m_v in modes.items():
        m_v["kernels"] = {k_k: k_v for k_k, k_v in m_v["kernels"].items() if k_k in kernels_limited}

    mode_name = "ou_bounded_seasonal"
    mode = modes[mode_name]
    k_name = "sin_rbf"
    config = mode["config"]
    config["meas_noise_var"] = 0.1
    plot_gp_regression_sample(mode_name=mode_name, k_name=k_name, kernel_sim=mode["kernels"][k_name],
                              data_fraction=0.2, **config)
    plot_evaluate_kernels(modes, data_fraction_list, n_samples=10)


