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


PARAM_NAMES = ["noise_level", "length_scale", "constant_value", "periodicity", "sigma_0", "nu"]

# Simple Kernels
simple_kernel_config = {
    "white": {"kernel": WhiteKernel, "params": {"noise_level": 1}, "bound_params": {}, "var": 1,
              "var_bounds": (0.1, 10)},
    "ou": {"kernel": Matern, "params": {"length_scale": 3, "nu": 0.5},
           "bound_params": {"length_scale_bounds": (0.1, 10)},
           "var": 5, "var_bounds": (0.1, 100)},
    "rbf_long": {"kernel": RBF, "params": {"length_scale": 50},
                 "bound_params": {"length_scale_bounds": (1, 200)},
                 "var": 5, "var_bounds": (0.1, 100)},
    "rbf_short": {"kernel": RBF, "params": {"length_scale": 3}, "bound_params": {"length_scale_bounds": (0.1, 20)},
                  "var": 1, "var_bounds": (0.1, 10)},
    "rbf_medium": {"kernel": RBF, "params": {"length_scale": 25},
                   "bound_params": {"length_scale_bounds": (1, 200)},
                   "var": 1, "var_bounds": (0.1, 10)},
    "sin_day": {"kernel": ExpSineSquared, "params": {
        "length_scale": 3, "periodicity": PERIOD_DAY}, "bound_params": {
        "periodicity_bounds": "fixed", "length_scale_bounds": (0.1, 20)},
                "var": 14**2, "var_bounds": (10, 1000)},
    # Note sample_ampl = sqrt(2*sample_var)
    # var of 100 leads to sample_var: 4.17, sample_ampl: 2.89 (overall max-min = 5.78)
    # var of 198 (=14.1**2) leads to sample_var: 13.28, sample_ampl: 5.15 (overall max-min = 5.78)

    # "sin_week": {"kernel": ExpSineSquared, "params": {
    #            "length_scale": 3, "periodicity": PERIOD_WEEK}, "bound_params": {"periodicity_bounds": "fixed"},
    #              "var": 5},
    "dot": {"kernel": DotProduct, "params": {"sigma_0": 0.001}, "bound_params": {"sigma_0_bounds": (0.0001, 0.01)},
            "var": 0.01, "var_bounds": (0.001, 1)}
           }

KERNELS = {}
OU_KERNELS = {}

for mode in ["fixed", "bounded", "unbounded"]:

    if mode == "unbounded":
        _simple_kernels = {k: ConstantKernel(constant_value=v["var"]) * v["kernel"](
            **{param: "fixed" for param in v["kernel"]().get_params().keys() if param == "periodicity_bounds"},
            **v["params"]) for k, v in simple_kernel_config.items()}
    if mode == "bounded":
        # Kernels with bounds defined in simple_kernel_config
        _simple_kernels = {k: ConstantKernel(
            constant_value=v["var"], constant_value_bounds=v["var_bounds"]) * v["kernel"](
            **v["bound_params"], **v["params"]) for k, v in simple_kernel_config.items()}
    if mode == "fixed":
        # Fixed Kernels, bounds form simple_kernel_config are ignored
        _simple_kernels = {k: ConstantKernel(constant_value=v["var"], constant_value_bounds="fixed") * v["kernel"](
                **{param: "fixed" for param in v["kernel"]().get_params().keys() if "bounds" in param},
                **v["params"]) for k, v in simple_kernel_config.items()}

    _combination_kernels = {
        "sinrbf": _simple_kernels["sin_day"] * _simple_kernels["rbf_long"],
        "sinrbf_rbf":  _simple_kernels["sin_day"] * _simple_kernels["rbf_medium"] +
                       _simple_kernels["rbf_long"],
                            "sin_rbf": _simple_kernels["sin_day"] + _simple_kernels["rbf_long"]
                            }
    _kernels = {**_simple_kernels, **_combination_kernels}

    ou_kernels = {k: (_kernels["ou"] + v if k != "ou" else v)
                  for k, v in _kernels.items()}

    KERNELS[mode] = _kernels
    OU_KERNELS[mode] = ou_kernels


# measuring time in hours
@dataclass()
class GPSimulatorConfig():

    n_days: int = 7
    samples_per_hour: int = 10
    meas_noise_var: float = 62 / 2  # Meas noise std = 7.9, leads to noise_var=62.
    # Var(Noise_CUFF - Noise_Aktiia) = Var(Noise_CUFF) + Var(Noise_Aktiia) - 2COV(Noise_CUFF - Noise_Aktiia)
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

    simulation_config_keys = ["mean_f", "x", "meas_noise_var", "kernel_sim", "data_fraction_weights"]

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
        if self.kernel_sim_name == "sin_rbf":
            return self.ou_kernel + self.sin_kernel + self.rbf_kernel

        elif self.kernel_sim_name == "sinrbf_rbf":
            return self.ou_kernel + self.sin_kernel * self.rbf_kernel + self.rbf_kernel

        else:
            raise ValueError(f"{self.kernel_sim_name=} does not exist")

    @property
    def x(self):
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


