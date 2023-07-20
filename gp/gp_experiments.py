import copy

from gp.gp_simulator import GPSimulator, GPSimulationEvaluator
from gp.simulate_gp_config import OU_KERNELS, base_config
from gp.post_sim_analysis import perf_plot, perf_plot_split
from logging import getLogger
from log_setup import setup_logging
import numpy as np
import pandas as pd
from constants.constants import get_output_path
from functools import partial
from datetime import datetime


logger = getLogger(__name__)

MODES = {
    "ou_bounded": {"kernels": OU_KERNELS["bounded"], "config": {"normalize_y": False, **base_config}},
    "ou_bounded_seasonal": {"kernels": OU_KERNELS["bounded"],
                            "config": {"normalize_y": False, "data_fraction_weights": "seasonal",
                                       **base_config}}}


def evaluate_data_fraction(mode_name, mode_config, data_fraction=(0.1, 0.2, 0.4),
                           n_samples=100, experiment_name="test"):
    mode_config = copy.copy(mode_config)
    meas_noise_var = mode_config["config"].pop("meas_noise_var")
    experiment_output_path = get_output_path(experiment_name=experiment_name)
    target_measures_path = experiment_output_path / f"target_measures_eval.csv"
    for k_name, k in mode_config["kernels"].items():
        session_name = f"{mode_name}_{k_name}"
        kernel, nv = GPSimulator.get_normalized_kernel(kernel=k, meas_noise_var=meas_noise_var)
        output_path_gp_sim = partial(get_output_path, session_name=session_name,
                                     experiment_name=experiment_name)
        for frac in data_fraction:
            logger.info(f"Simulation started for {experiment_name=}.{session_name=}, {frac=} and {nv=} ")
            rng = np.random.default_rng(15)
            simulator = GPSimulationEvaluator(
                output_path=output_path_gp_sim, rng=rng, kernel_sim=kernel, data_fraction=frac, normalize_kernel=False,
                meas_noise_var=nv, **mode_config["config"])
            simulator.plot_gp_regression_sample(nplots=1)
            simulator.evaluate()
            simulator.evaluate_target_measures()
            eval_dict, measure_sum_df = simulator.evaluate_multisample(n_samples)

            for k, v in simulator.current_init_kwargs.items():
                if not isinstance(v, np.ndarray):
                    measure_sum_df[k] = v

            if not (target_measures_path).exists():
                measure_sum_df.to_csv(target_measures_path, index=None)
            else:
                measure_sum_df.to_csv(target_measures_path, mode='a', header=False, index=None)

            for k, v in eval_dict.items():
                df = pd.DataFrame([v])
                df["mode"] = mode_name
                df["kernel_name"] = k_name
                if not (experiment_output_path / f"{k}.csv").exists():
                    df.to_csv(experiment_output_path / f"{k}.csv", index=None)
                else:
                    df.to_csv(experiment_output_path / f"{k}.csv", mode='a', header=False, index=None)

    return df


def evaluate_data_fraction_modes(modes, data_fraction=(0.1, 0.2, 0.4), meas_noise_var=(None,),
                                 n_samples=100, experiment_name="test"):
    experiment_output_path = get_output_path(experiment_name=experiment_name)

    for nv in meas_noise_var:
        for mode_name, mode_config in modes.items():
            if nv is not None:
                mode_config["config"]["meas_noise_var"] = nv
            evaluate_data_fraction(mode_name, mode_config, data_fraction=data_fraction, n_samples=n_samples,
                                   experiment_name=experiment_name)

    # for split in ["overall", "train", "test"]:
    #     perf_plot(split=split, file_path=experiment_output_path)


def get_limited_modes(kernels_limited=None, modes_limited=None):
    modes = copy.deepcopy(MODES)
    if modes_limited:
        modes = {k: v for k, v in modes.items() if k in modes_limited}
    if kernels_limited:
        for m_k, m_v in modes.items():
            m_v["kernels"] = {k_k: k_v for k_k, k_v in m_v["kernels"].items() if k_k in kernels_limited}
    return modes


def plot_sample(k_name="sin_rbf", mode_name="ou_bounded_seasonal", data_fraction=0.2,
                normalize_kernel=True, experiment_name="test_single_sample", rng=None, nplots=1):
    mode = MODES[mode_name]
    session_name = f"{mode_name}_{k_name}"
    output_path_gp_sim = partial(get_output_path, session_name=session_name, experiment_name=experiment_name)
    simulator = GPSimulationEvaluator(
        output_path=output_path_gp_sim, kernel_sim=mode["kernels"][k_name], data_fraction=data_fraction,
        normalize_kernel=normalize_kernel, **mode["config"], rng=rng)
    simulator.plot_gp_regression_sample(nplots=nplots)


if __name__ == "__main__":
    setup_logging()

    kernels_limited = ["sin_rbf"]
    modes_limited = ["ou_bounded", "ou_bounded_seasonal"]
    modes = get_limited_modes(kernels_limited=kernels_limited, modes_limited=modes_limited)

    rng = np.random.default_rng(18)
    plot_sample(normalize_kernel=False, rng=rng, experiment_name="data_fraction_test4", nplots=1)
    # evaluate_data_fraction(mode_name="ou_bounded", mode_config=modes["ou_bounded"],
    #                        n_samples=2, experiment_name="data_fraction_test_2")
    # evaluate_data_fraction_modes(modes, n_samples=100, experiment_name="data_fraction_test_ou5")


