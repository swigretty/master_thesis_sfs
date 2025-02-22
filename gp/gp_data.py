from dataclasses import dataclass, asdict
from functools import cached_property
from evalutation_utils import calculate_ci
import numpy as np
import pandas as pd


@dataclass
class GPData():
    """
    Class storing all the relevant information to describe a
    Gaussian Process (GP) as a finite set of random variables
    and to be used for GP regression.

    Assuming, y(x) ~ GP(m(x), k(x, x')) we have:

    y_mean = [m(x_1), ..., m(x_n)]
    y_cov = [[k(x_1, x_1) ... k(x_1, x_n],
             .
             .
             .
             [k(x_n, x_1) ... k(x_n, x_n]
             ]


    At the very least you need to provide either y or y_mean.
    In most cases you will skip y, which is just a realization (sample) of
    the GP described by y_mean and y_cov.

    """
    # TODO rename to sample,mean,cov (since y, y_mean and, y_cov is confusing)
    # A sample from the GP
    y: np.array = None
    # design matrix (or independent, explanatory variables). (time)
    x: np.array = None
    y_mean: np.array = None  # The mean function of the GP evaluated at input x
    y_cov: np.array = None  # The cov function values evaluated at the input x

    def __post_init__(self):
        if self.y is None:
            if self.y_mean is None:
                raise ValueError("Either y_mean or y need to be specified")

        self.index = 0
        self.check_dimensions()

    def check_dimensions(self):
        if self.x is not None and self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        if self.y is not None and self.y.ndim == 2:
            self.y = self.y.reshape(-1)
        if self.y_mean is not None and self.y_mean.ndim == 2:
            self.y_mean = self.y_mean.reshape(-1)

        arrays = [self.x, self.y, self.y_mean]
        arrays = [arr for arr in arrays if arr is not None]
        assert all([arrays[i].shape[0] == arrays[i+1].shape[0] for i in range(
            len(arrays)-1)])
        if self.y_cov is not None:
            assert self.y_cov.shape == (len(self.x), len(self.x))

    def __len__(self):
        return len(self.x)

    def to_df(self):
        df = pd.DataFrame({k: v for k, v in asdict(self).items() if k not in ["y_cov", "x"] and v is not None})
        df["y_var"] = np.diag(self.y_cov)
        if self.x.shape[1] > 1:
            for i in self.x.shape[1]:
                df[f"x{i}"] = self.x[:, i]
        else:
            df["x"] = self.x.reshape(-1)
        return df

    @classmethod
    def from_df(cls, df):
        return cls(**df.to_dict())

    def get_field_item(self, field, idx):
        value = getattr(self, field)
        if value is None:
            return value
        if field == "y_cov":
            if isinstance(idx, int):
                return value[idx, idx]
            return np.array([[self.y_cov[io, ii] for ii in idx] for io in idx])
        return value[idx]

    def __getitem__(self, idx):
        return self.__class__(**{k: self.get_field_item(k, idx) for k in
                                 asdict(self).keys()})

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self):
            self.index = 0
            raise StopIteration
        self.index += 1
        return self[self.index - 1]

    def __add__(self, value):
        new_fields = {k: (v + value if k in ["y_mean", "y"] and v is not None
                          else v) for k, v in asdict(self).items()}
        return self.__class__(**new_fields)

    def __sub__(self, value):
        new_fields = {k: (v - value if k in ["y_mean", "y"] and v is not None
                          else v) for k, v in asdict(self).items()}
        return self.__class__(**new_fields)

    def __mul__(self, value):
        new_fields = {k: (v * value if k in ["y_mean", "y"] and v is not None
                          else v) for k, v in asdict(self).items()}
        new_fields["y_cov"] *= value**2
        return self.__class__(**new_fields)

    @property
    def y_var(self):
        if self.y_cov is None:
            return None
        return np.diag(self.y_cov)

    @property
    def y_std(self):
        if self.y_var is None:
            return None
        return np.sqrt(self.y_var)

    def calculate_ci_row(self, row):
        return calculate_ci(np.sqrt(row["y_var"]), row["y_mean"])

    @cached_property
    def ci(self):
        df = self.to_df()
        ci = pd.DataFrame(df.apply(lambda row: self.calculate_ci_row(row),
                                   axis=1).to_list())
        return ci

    def get_samples(self, n_samples=10, rng=None) -> list:
        samples_list = []
        if rng is None:
            rng = np.random.default_rng()
        y_samples = rng.multivariate_normal(self.y_mean, self.y_cov, n_samples)
        kwargs = asdict(self)
        for s in y_samples:
            kwargs["y"] = s
            samples_list.append(self.__class__(**kwargs))
        return samples_list


