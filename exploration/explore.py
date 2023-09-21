import numpy as np


def get_red_idx(n, data_fraction=0.3, weights=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(11)
    if data_fraction == 1:
        return range(n)
    k = int(n * data_fraction)
    if weights is not None:
        weights = weights/sum(weights)
    return sorted(rng.choice(range(n), size=k, replace=False, p=weights))

