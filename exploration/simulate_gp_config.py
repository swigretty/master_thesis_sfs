from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
from functools import partial
from dataclasses import dataclass, asdict
from logging import getLogger
import numpy as np
from log_setup import setup_logging

logger = getLogger(__name__)


PERIOD_DAY = 24
PERIOD_WEEK = 7 * PERIOD_DAY


def mean_fun_const(x):
    # 110 to 130 (healthy range)
    # physiological:  60 to 300
    return 120


# measuring time in hours
@dataclass()
class GPSimulatorConfig():

    n_days: int = 3
    samples_per_hour: int = 10

    meas_noise: int = 0
    mean_f: callable = mean_fun_const

    simulation_config_keys = ["meas_noise", "mean_f", "x"]

    @property
    def x(self):
        return np.linspace(0, PERIOD_DAY * self.n_days, PERIOD_DAY * self.n_days * self.samples_per_hour)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.simulation_config_keys}


# Simple Kernels
simple_kernel_config = {
    "white": {"kernel": WhiteKernel, "params": {"noise_level": 1}, "bound_params": {}, "scale": 1},
    "ou": {"kernel": Matern, "params": {"length_scale": 3, "nu": 0.5}, "bound_params": {}, "scale": 1},
    "rbf_long": {"kernel": RBF, "params": {"length_scale": 50}, "bound_params": {}, "scale": 1},
    "rbf_short": {"kernel": RBF, "params": {"length_scale": 3}, "bound_params": {}, "scale": 1},
    "sin_day": {"kernel": ExpSineSquared, "params": {
        "length_scale": 3, "periodicity": PERIOD_DAY}, "bound_params": {"periodicity_bounds": "fixed"}, "scale": 10},
    "sin_week": {"kernel": ExpSineSquared, "params": {
               "length_scale": 3, "periodicity": PERIOD_WEEK}, "bound_params": {"periodicity_bounds": "fixed"},
                 "scale": 5},
    "dot": {"kernel": DotProduct, "params": {"sigma_0": 0}, "bound_params": {}, "scale": 0.01}
           }

# Fixed Kernels, bounds form simple_kernel_config are ignored
_simple_kernels_fixed = {k: ConstantKernel(constant_value=v["scale"], constant_value_bounds="fixed") * v["kernel"](
    **{param: "fixed" for param in v["kernel"]().get_params().keys() if "bounds" in param},
    **v["params"]) for k, v in simple_kernel_config.items()}
_kernels_fixed = {"sinrbf": _simple_kernels_fixed["sin_day"] * _simple_kernels_fixed["rbf_long"],
                  **_simple_kernels_fixed}

# Kernels with bounds defined in simple_kernel_config
_simple_kernels = {k: ConstantKernel(constant_value=v["scale"]) * v["kernel"](
    **v["bound_params"], **v["params"]) for k, v in simple_kernel_config.items()}
_kernels = {"sinrbf": _simple_kernels["sin_day"] * _simple_kernels["rbf_long"], **_simple_kernels}

ou_kernels_fixed = {k: _kernels_fixed["ou"] + v for k, v in _kernels_fixed.items()}
ou_kernels = {k: _kernels["ou"] + v for k, v in _kernels.items()}

base_config = GPSimulatorConfig().to_dict()


if __name__ == "__main__":
    setup_logging()
    logger.info(ou_kernels_fixed)
    # logger.info(base_config)


