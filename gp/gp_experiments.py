"""
This modules stores functionality to run the main experiments
to assess performance of GP regression and some baseline methods
in estimating simulated BP values.
"""
import copy
from gp.gp_simulator import GPSimulator, GPSimulationEvaluator
from gp.simulate_gp_config import GPSimulatorConfig
from logging import getLogger
from log_setup import setup_logging
import numpy as np
import pandas as pd
from constants.constants import get_output_path
from functools import partial

logger = getLogger(__name__)


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


def evaluate_data_fraction(
        mode_config: GPSimulatorConfig,
        data_fraction: tuple = (0.05, 0.1, 0.2, 0.4),
        n_samples: int = 100, experiment_name: str = "test",
        normalize_kernel: bool = False,
        normalize_y: bool = True, seed: int = 15):
    """
    Run simulation for different data fractions. Makes sure that the same
    samples are drawn from the GP for every data fraction.

    Keyword arguments:

    normalyze_y: This will mean center and scale to unit variance the simulated
    data produced before fitting the GP regression. This is important for
    the optimizer used when optimizing the hyperparameters. If set to False
    results will generally be worse. However, when set to True
    the found hyperparameters during GP regression cannot be compared to the
    ones used during simulation anymore.

    normalize_kernel: Experimental, you generally want to set this to False.
    If set to True this will modify the simulation kernel specified in
    mode_config.kernel_sim such that it will produce sample paths
    that have close to unit variance. You may want to set to True if you
    want to be able to compare the simulated kernel to the fitted kernel
    hyperparameters. However, in such cases you should set normalize_y to False.

    """

    mode_config_orig = mode_config.to_dict()
    mode_config_norm = copy.copy(mode_config_orig)

    experiment_output_path = get_output_path(experiment_name=experiment_name)
    target_measures_path = experiment_output_path / f"target_measures_eval.csv"

    session_name = f"{mode_config.session_name}"
    output_path_gp_sim = partial(get_output_path, session_name=session_name,
                                 experiment_name=experiment_name)

    orig_scale = get_sample_path_variance_kernel(mode_config.kernel_sim,
                                                 mode_config.x)
    orig_scale += mode_config.meas_noise_var
    logger.info(f"{orig_scale=}")

    if normalize_kernel:
        (mode_config_norm["kernel_sim"], mode_config_norm["meas_noise_var"],
         scale) = GPSimulator.get_normalized_kernel(
            kernel=mode_config.kernel_sim,
            meas_noise_var=mode_config.meas_noise_var)

    for frac in data_fraction:
        logger.info(f"Simulation started for {experiment_name=}, "
                    f"{session_name=} and {frac=}")
        rng = np.random.default_rng(seed)
        simulator = GPSimulationEvaluator(
            output_path=output_path_gp_sim, rng=rng, data_fraction=frac,
            normalize_kernel=False,
            normalize_y=normalize_y, **mode_config_norm)
        simulator.plot_gp_regression_sample(nplots=1)
        simulator.evaluate()
        simulator.evaluate_target_measures()
        eval_dict, measure_sum_df = simulator.evaluate_multisample(n_samples)

        for k, v in simulator.current_init_kwargs.items():
            if not isinstance(v, np.ndarray):
                measure_sum_df[k] = v

        # Write the evaluation summary for the target measures to csv
        if not (target_measures_path).exists():
            measure_sum_df.to_csv(target_measures_path, index=None)
        else:
            measure_sum_df.to_csv(target_measures_path, mode='a', header=False,
                                  index=None)

        # General performanc metrics only calculated for the GP regression
        # Stores test, train and overall performance in different tables
        for k, v in eval_dict.items():
            df = pd.DataFrame([v])
            df["session_name"] = session_name
            if not (experiment_output_path / f"{k}.csv").exists():
                df.to_csv(experiment_output_path / f"{k}.csv", index=None)
            else:
                df.to_csv(experiment_output_path / f"{k}.csv", mode='a',
                          header=False, index=None)

    return df


def plot_sample(normalize_kernel: bool = False,
                experiment_name: str = "test_single_sample",
                rng: np.random.Generator = None, nplots: int = 1,
                config: GPSimulatorConfig = GPSimulatorConfig(),
                data_fraction: float = 0.2,
                normalize_y: bool = True, plot_method: str = None):

    output_path_gp_sim = partial(get_output_path,
                                 session_name=config.session_name,
                                 experiment_name=experiment_name)
    simulator = GPSimulationEvaluator(
        output_path=output_path_gp_sim, normalize_kernel=normalize_kernel,
        rng=rng, data_fraction=data_fraction,
        normalize_y=normalize_y, **config.to_dict())
    simulator.plot_gp_regression_sample(nplots=nplots, plot_method=plot_method)


MODES = [
        GPSimulatorConfig(kernel_sim_name="sin_rbf",
                          session_name="sin_rbf_default"),
        GPSimulatorConfig(kernel_sim_name="sin_rbf",
                          data_fraction_weights=lambda x: x ** 1,
                          session_name="sin_rbf_seasonal_default"),
        GPSimulatorConfig(kernel_sim_name="sin_rbf",
                          data_fraction_weights=lambda x: x ** 2,
                          session_name="sin_rbf_seasonal_extreme")
             ]


if __name__ == "__main__":
    setup_logging()

    modes = MODES

    # Run main Simulation Experiment accross different data fractions
    # and sampling patterns (modes).
    experiment_name = "my_experiment"
    for mode in modes:
        evaluate_data_fraction(mode,
                               n_samples=100,
                               experiment_name=experiment_name,
                               normalize_kernel=False,
                               normalize_y=True, seed=15)

    # Generate plots visualizing predictions of regression methods
    # fitted to data based on different samples drawn form the true GP
    experiment_name = "my_plots"
    for datafrac in [0.05, 0.1, 0.2, 0.4]:
        for mode in modes:
            mode.session_name = f"{mode.session_name}_{datafrac}"
            rng = np.random.default_rng(18)
            plot_sample(normalize_kernel=False, rng=rng,
                        experiment_name=experiment_name, nplots=10,
                        config=mode,
                        data_fraction=datafrac,
                        normalize_y=True)
