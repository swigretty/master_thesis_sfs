import copy

from gp.gp_simulator import GPSimulator, GPSimulationEvaluator
from gp.simulate_gp_config import base_config, GPSimulatorConfig
from gp.post_sim_analysis import perf_plot, perf_plot_split
from logging import getLogger
from log_setup import setup_logging
import numpy as np
import pandas as pd
from constants.constants import get_output_path
from functools import partial
from datetime import datetime


logger = getLogger(__name__)

# MODES = {
#     "ou_bounded_sin_rbf": partial(GPSimulatorConfig, kernel_sim_name="sin_rbf"),
#     "ou_bounded_seasonal": partial(GPSimulatorConfig, kernel_sim_name="sin_rbf",
#                                    data_fraction_weights=lambda x: x ** 1),
#     # "ou_bounded_seasonal_extreme": {"kernels": OU_KERNELS["bounded"],
#     #                         "config": {"normalize_y": False, "data_fraction_weights": lambda x: x**2,
#     #                                    **base_config}},
#     # "ou_bounded_seasonal_mild": {"kernels": OU_KERNELS["bounded"],
#     #                                 "config": {"normalize_y": False, "data_fraction_weights": lambda x: x**0.5,
#     #                                            **base_config}},
#
# }


def evaluate_data_fraction(mode_config, data_fraction=(0.1, 0.2, 0.4),
                           n_samples=100, experiment_name="test"):
    mode_config_orig = mode_config.to_dict()
    mode_config_norm = copy.copy(mode_config_orig)

    experiment_output_path = get_output_path(experiment_name=experiment_name)
    target_measures_path = experiment_output_path / f"target_measures_eval.csv"

    session_name = f"{mode_config.kernel_sim_name}"
    output_path_gp_sim = partial(get_output_path, session_name=session_name,
                                 experiment_name=experiment_name)

    mode_config_norm["kernel_sim"], mode_config_norm["meas_noise_var"] = GPSimulator.get_normalized_kernel(
        kernel=mode_config.kernel_sim, meas_noise_var=mode_config.meas_noise_var)

    for frac in data_fraction:
        logger.info(f"Simulation started for {experiment_name=}, {session_name=} and {frac=}")
        rng = np.random.default_rng(15)
        simulator = GPSimulationEvaluator(
            output_path=output_path_gp_sim, rng=rng, data_fraction=frac, normalize_kernel=False, **mode_config_norm)
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
            df["session_name"] = session_name
            if not (experiment_output_path / f"{k}.csv").exists():
                df.to_csv(experiment_output_path / f"{k}.csv", index=None)
            else:
                df.to_csv(experiment_output_path / f"{k}.csv", mode='a', header=False, index=None)

    return df


def evaluate_data_fraction_modes(modes, data_fraction=(0.1, 0.2, 0.4), meas_noise_var=(None,),
                                 n_samples=100, experiment_name="test"):
    experiment_output_path = get_output_path(experiment_name=experiment_name)

    for nv in meas_noise_var:
        for mode_config in modes():
            if nv is not None:
                mode_config = mode_config(meas_noise_var=nv)
            else:
                mode_config = mode_config()
            evaluate_data_fraction(mode_config, data_fraction=data_fraction, n_samples=n_samples,
                                   experiment_name=experiment_name)

    # for split in ["overall", "train", "test"]:
    #     perf_plot(split=split, file_path=experiment_output_path)


# def get_limited_modes(kernels_limited=None, modes_limited=None):
#     modes = copy.deepcopy(MODES)
#     if modes_limited:
#         modes = {k: v for k, v in modes.items() if k in modes_limited}
#     if kernels_limited:
#         for m_k, m_v in modes.items():
#             m_v["kernels"] = {k_k: k_v for k_k, k_v in m_v["kernels"].items() if k_k in kernels_limited}
#     return modes


def plot_sample(normalize_kernel=True, experiment_name="test_single_sample", rng=None,
                nplots=1, config=GPSimulatorConfig(), data_fraction=0.2):
    output_path_gp_sim = partial(get_output_path, session_name=config.kernel_sim_name, experiment_name=experiment_name)
    simulator = GPSimulationEvaluator(
        output_path=output_path_gp_sim, normalize_kernel=normalize_kernel, rng=rng, data_fraction=data_fraction,
        **config.to_dict())
    simulator.plot_gp_regression_sample(nplots=nplots)


if __name__ == "__main__":
    setup_logging()

    modes = [partial(GPSimulatorConfig, kernel_sim_name="sin_rbf"),
             partial(GPSimulatorConfig, kernel_sim_name="sin_rbf", data_fraction_weights=lambda x: x ** 1)]

    rng = np.random.default_rng(18)
    experiment_name = "data_fraction_default"

    plot_sample(normalize_kernel=False, rng=rng, experiment_name=experiment_name, nplots=1,
                config=GPSimulatorConfig(kernel_sim_name="sin_rbf"), data_fraction=0.2)
    evaluate_data_fraction(GPSimulatorConfig(kernel_sim_name="sin_rbf"),
                           experiment_name=experiment_name)
    # evaluate_data_fraction_modes(modes, n_samples=2, experiment_name="default_modes")


