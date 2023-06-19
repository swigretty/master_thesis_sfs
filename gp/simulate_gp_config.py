from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic, \
    Matern, ConstantKernel, DotProduct
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

    n_days: int = 7
    samples_per_hour: int = 10

    meas_noise: int = 0
    mean_f: callable = mean_fun_const

    simulation_config_keys = ["meas_noise", "mean_f", "x"]

    @property
    def x(self):
        return np.linspace(0, PERIOD_DAY * self.n_days, PERIOD_DAY * self.n_days * self.samples_per_hour)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.simulation_config_keys}


PARAM_NAMES = ["noise_level", "length_scale", "constant_value", "periodicity", "sigma_0", "nu"]

# Simple Kernels
simple_kernel_config = {
    "white": {"kernel": WhiteKernel, "params": {"noise_level": 1}, "bound_params": {}, "scale": 1, "scale_bounds": (0.1, 10)},
    "ou": {"kernel": Matern, "params": {"length_scale": 3, "nu": 0.5}, "bound_params": {"length_scale_bounds": (1, 10)},
           "scale": 1, "scale_bounds": (0.1, 10)},
    "rbf_long": {"kernel": RBF, "params": {"length_scale": 50}, "bound_params": {"length_scale_bounds": (1, 200)},
                 "scale": 1, "scale_bounds": (0.1, 10)},
    "rbf_short": {"kernel": RBF, "params": {"length_scale": 3}, "bound_params": {"length_scale_bounds": (1, 200)},
                  "scale": 1, "scale_bounds": (0.1, 10)},
    "rbf_medium": {"kernel": RBF, "params": {"length_scale": 25}, "bound_params": {"length_scale_bounds": (1, 200)},
                  "scale": 1, "scale_bounds": (0.1, 10)},
    "sin_day": {"kernel": ExpSineSquared, "params": {
        "length_scale": 3, "periodicity": PERIOD_DAY}, "bound_params": {"periodicity_bounds": "fixed"},
                "scale": 10, "scale_bounds": (1, 100)},
    # "sin_week": {"kernel": ExpSineSquared, "params": {
    #            "length_scale": 3, "periodicity": PERIOD_WEEK}, "bound_params": {"periodicity_bounds": "fixed"},
    #              "scale": 5},
    "dot": {"kernel": DotProduct, "params": {"sigma_0": 0.001}, "bound_params": {"sigma_0_bounds": (0.0001, 0.01)},
            "scale": 0.01, "scale_bounds": (0.001, 1)}
           }

KERNELS = {}
OU_KERNELS = {}

for mode in ["fixed", "bounded", "unbounded"]:

    if mode == "unbounded":
        _simple_kernels = {k: ConstantKernel(constant_value=v["scale"]) * v["kernel"](
            **{param: "fixed" for param in v["kernel"]().get_params().keys() if param == "periodicity_bounds"},
            **v["params"]) for k, v in simple_kernel_config.items()}
    if mode == "bounded":
        # Kernels with bounds defined in simple_kernel_config
        _simple_kernels = {k: ConstantKernel(
            constant_value=v["scale"], constant_value_bounds=v["scale_bounds"]) * v["kernel"](
            **v["bound_params"], **v["params"]) for k, v in simple_kernel_config.items()}
    if mode == "fixed":
        # Fixed Kernels, bounds form simple_kernel_config are ignored
        _simple_kernels = {k: ConstantKernel(constant_value=v["scale"], constant_value_bounds="fixed") * v["kernel"](
                **{param: "fixed" for param in v["kernel"]().get_params().keys() if "bounds" in param},
                **v["params"]) for k, v in simple_kernel_config.items()}

    _combination_kernels = {"sinrbf": _simple_kernels["sin_day"] * _simple_kernels["rbf_long"],
                            "sinrbf_rbf":  _simple_kernels["sin_day"] * _simple_kernels["rbf_medium"] +
                                           _simple_kernels["rbf_long"],
                            "sin_rbf": _simple_kernels["sin_day"] + _simple_kernels["rbf_long"]
                            }
    _kernels = {**_simple_kernels, **_combination_kernels}

    ou_kernels = {k: (_kernels["ou"] + v + WhiteKernel(noise_level=0.0001, noise_level_bounds="fixed") if
                      k != "ou" else v + WhiteKernel(noise_level=0.0001, noise_level_bounds="fixed"))
                  for k, v in _kernels.items()}

    KERNELS[mode] = _kernels
    OU_KERNELS[mode] = ou_kernels

base_config = GPSimulatorConfig().to_dict()


if __name__ == "__main__":
    setup_logging()
    logger.info(ou_kernels_fixed)
    # logger.info(base_config)


