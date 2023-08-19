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


def get_info(sample):
    return {"var": np.var(sample), "max-min": np.max(sample) - np.min(sample)}


def get_sample_path_variance_kernel(kernel, t, nsim=1000):
    mu = np.zeros(len(t))
    K = kernel(np.array(t).reshape(-1, 1))
    sim_info = []
    y_sim = np.random.multivariate_normal(mean=mu, cov=K, size=nsim)
    for sample in y_sim:
        sim_info.append(get_info(sample))
    sim_info_mean = pd.DataFrame(sim_info).mean().to_dict()
    return sim_info_mean["var"]


def evaluate_data_fraction(mode_config, data_fraction=(0.05, 0.1, 0.2, 0.4),
                           n_samples=100, experiment_name="test", normalize_kernel=False,
                           normalize_y=True, only_var=False):
    mode_config_orig = mode_config.to_dict()
    mode_config_norm = copy.copy(mode_config_orig)

    experiment_output_path = get_output_path(experiment_name=experiment_name)
    target_measures_path = experiment_output_path / f"target_measures_eval.csv"

    session_name = f"{mode_config.session_name}"
    output_path_gp_sim = partial(get_output_path, session_name=session_name,
                                 experiment_name=experiment_name)

    orig_scale = get_sample_path_variance_kernel(mode_config.kernel_sim, mode_config.x)
    orig_scale += mode_config.meas_noise_var
    logger.info(f"{orig_scale=}")

    if normalize_kernel:
        mode_config_norm["kernel_sim"], mode_config_norm["meas_noise_var"] = GPSimulator.get_normalized_kernel(
            kernel=mode_config.kernel_sim, meas_noise_var=mode_config.meas_noise_var)

    for frac in data_fraction:
        logger.info(f"Simulation started for {experiment_name=}, {session_name=} and {frac=}")
        rng = np.random.default_rng(15)
        simulator = GPSimulationEvaluator(
            output_path=output_path_gp_sim, rng=rng, data_fraction=frac, normalize_kernel=False,
            normalize_y=normalize_y, **mode_config_norm)
        simulator.plot_gp_regression_sample(nplots=1)
        simulator.evaluate()
        simulator.evaluate_target_measures()
        eval_dict, measure_sum_df = simulator.evaluate_multisample(n_samples, only_var=only_var)

        if only_var:
            return

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


def evaluate_data_fraction_modes(modes, data_fraction=(0.05, 0.1, 0.2, 0.4), meas_noise_var=(None,),
                                 n_samples=100, experiment_name="test", normalize_kernel=False, normalize_y=True):
    experiment_output_path = get_output_path(experiment_name=experiment_name)

    for nv in meas_noise_var:
        for mode_config in modes:
            if nv is not None:
                mode_config = mode_config(meas_noise_var=nv)
            else:
                mode_config = mode_config()
            evaluate_data_fraction(mode_config, data_fraction=data_fraction, n_samples=n_samples,
                                   experiment_name=experiment_name, normalize_kernel=normalize_kernel,
                                   normalize_y=normalize_y)

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


def plot_sample(normalize_kernel=False, experiment_name="test_single_sample", rng=None,
                nplots=1, config=GPSimulatorConfig(), data_fraction=0.2, normalize_y=True, plot_method=None):
    output_path_gp_sim = partial(get_output_path, session_name=config.session_name, experiment_name=experiment_name)
    simulator = GPSimulationEvaluator(
        output_path=output_path_gp_sim, normalize_kernel=normalize_kernel, rng=rng, data_fraction=data_fraction,
        normalize_y=normalize_y, **config.to_dict())
    simulator.plot_gp_regression_sample(nplots=nplots, plot_method=plot_method)


if __name__ == "__main__":
    setup_logging()

    modes = [
        partial(GPSimulatorConfig, kernel_sim_name="sin_rbf",
                     session_name="sin_rbf_default"),
        partial(GPSimulatorConfig, kernel_sim_name="sin_rbf", data_fraction_weights=lambda x: x ** 1,
                 session_name="sin_rbf_seasonal_default"),
        partial(GPSimulatorConfig, kernel_sim_name="sin_rbf", data_fraction_weights=lambda x: x ** 2,
                 session_name="sin_rbf_seasonal_extreme")
             ]

    rng = np.random.default_rng(18)
    experiment_name = "new_measures_spline_fixed_no_transform_quantile"
    # for datafrac in [0.05, 0.1, 0.2, 0.4, 0.6]:
    #     plot_sample(normalize_kernel=False, rng=rng, experiment_name=experiment_name, nplots=3,
    #                 config=GPSimulatorConfig(kernel_sim_name="sin_rbf", session_name="10foldcv_sin_rbf_notperiodic"),
    #                 data_fraction=datafrac,
    #                 normalize_y=True, plot_method="plot_posterior_baseline")

    # evaluate_data_fraction(GPSimulatorConfig(kernel_sim_name="sin_rbf", session_name="sin_rbf"),
    #                        experiment_name=experiment_name, n_samples=2, data_fraction=(0.1, ),
    #                        normalize_kernel=False, normalize_y=True)
    evaluate_data_fraction_modes(modes, n_samples=100, experiment_name=experiment_name, normalize_y=True,
                                 normalize_kernel=False)


