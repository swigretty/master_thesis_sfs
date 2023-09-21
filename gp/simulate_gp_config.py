"""
GPSimulatorConfig class to construct configurations to be used for
gp.gp_simulator.GPSimulationEvaluator
"""
from dataclasses import dataclass
from logging import getLogger
import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ExpSineSquared, Matern, ConstantKernel, DotProduct)

from log_setup import setup_logging

logger = getLogger(__name__)


PERIOD_DAY = 24
PERIOD_WEEK = 7 * PERIOD_DAY


def mean_fun_const(x):
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    return 120


PARAM_NAMES = ["noise_level", "length_scale", "constant_value",
               "periodicity", "sigma_0", "nu"]

# Simple Kernels
simple_kernel_config = {
    "white": {"kernel": WhiteKernel, "params": {"noise_level": 1},
              "bound_params": {}, "var": 1,
              "var_bounds": (0.1, 10)},
    "ou": {"kernel": Matern, "params": {"length_scale": 3, "nu": 0.5},
           "bound_params": {"length_scale_bounds": (0.1, 10)},
           "var": 5, "var_bounds": (0.1, 100)},
    "rbf_long": {"kernel": RBF, "params": {"length_scale": 50},
                 "bound_params": {"length_scale_bounds": (1, 200)},
                 "var": 5, "var_bounds": (0.1, 100)},
    "rbf_short": {"kernel": RBF, "params": {"length_scale": 3},
                  "bound_params": {"length_scale_bounds": (0.1, 20)},
                  "var": 1, "var_bounds": (0.1, 10)},
    "rbf_medium": {"kernel": RBF, "params": {"length_scale": 25},
                   "bound_params": {"length_scale_bounds": (1, 200)},
                   "var": 1, "var_bounds": (0.1, 10)},
    "sin_day": {"kernel": ExpSineSquared, "params": {
        "length_scale": 3, "periodicity": PERIOD_DAY}, "bound_params": {
        "periodicity_bounds": "fixed", "length_scale_bounds": (0.1, 20)},
                "var": 14**2, "var_bounds": (10, 1000)},
    # Note sample_ampl = sqrt(2*sample_var)
    # var of 100 leads to sample_var: 4.17,
    # sample_ampl: 2.89 (overall max-min = 5.78)
    # var of 198 (=14.1**2) leads to sample_var: 13.28,
    # sample_ampl: 5.15 (overall max-min = 5.78)
    "dot": {"kernel": DotProduct, "params": {"sigma_0": 0.001},
            "bound_params": {"sigma_0_bounds": (0.0001, 0.01)},
            "var": 0.01, "var_bounds": (0.001, 1)}
           }


@dataclass()
class GPSimulatorConfig():
    """
    This class can be used to obtain the configuration for
    gp.gp_simulator.GPSimulationEvaluator.
    Example usage:
    gpe = gp.gp_simulator.GPSimulationEvaluator(
                        **GPSimulatorConfig().to_dict())
    """

    n_days: int = 7
    samples_per_hour: int = 10

    # Measurement noise variance is half of Var(Noise_CUFF - Noise_Aktiia) = 62
    meas_noise_var: float = 62 / 2
    mean_f: callable = mean_fun_const
    kernel_sim_name: str = "sin_rbf"

    sin_var: float = 14**2
    sin_var_bounds: tuple = (10, 1000)
    _sin_kernel = simple_kernel_config["sin_day"]

    ou_var: float = 5
    ou_var_bounds: tuple = (0.5, 50)
    _ou_kernel = simple_kernel_config["ou"]

    rbf_var: float = 5
    rbf_var_bounds: float = (1, 10)
    _rbf_kernel = simple_kernel_config["rbf_long"]

    data_fraction_weights: callable = None

    session_name: str = None

    kwargs: dict = None

    simulation_config_keys = ["mean_f", "x", "meas_noise_var", "kernel_sim",
                              "data_fraction_weights"]

    def __post_init__(self):
        for k_name in ["sin", "rbf", "ou"]:
            base_config = getattr(self, f"_{k_name}_kernel")
            kernel = ConstantKernel(
                constant_value=getattr(self, f"{k_name}_var"),
                constant_value_bounds=getattr(
                    self, f"{k_name}_var_bounds")) * base_config["kernel"](
                **base_config["bound_params"], **base_config["params"])

            setattr(self, f"{k_name}_kernel", kernel)
        if self.session_name is None:
            self.session_name = self.kernel_sim_name
            if self.data_fraction_weights is not None:
                self.session_name = f"{self.session_name}_seasonal"

    @property
    def kernel_sim(self):
        """
        The kernel to be used for simulating the true BP values
        """
        if self.kernel_sim_name == "sin_rbf":
            return self.ou_kernel + self.sin_kernel + self.rbf_kernel

        # If you want to simulate with an evolving seasonal pattern
        elif self.kernel_sim_name == "sinrbf_rbf":
            return (self.ou_kernel + self.sin_kernel * self.rbf_kernel +
                    self.rbf_kernel)

        else:
            raise ValueError(f"{self.kernel_sim_name=} does not exist")

    @property
    def x(self):
        """
        The time points, at which we are going to simulate and predict BP
        values. It spans one week with 10 points per hour and is measured
        in hours.
        """
        return np.linspace(0, PERIOD_DAY * self.n_days,
                           PERIOD_DAY * self.n_days * self.samples_per_hour)

    def to_dict(self):
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        base_dict = {k: getattr(self, k) for k in self.simulation_config_keys}
        return {**base_dict, **kwargs}


base_config = GPSimulatorConfig().to_dict()


if __name__ == "__main__":
    setup_logging()


