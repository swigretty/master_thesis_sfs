from sklearn.gaussian_process.kernels import RBF,  WhiteKernel, ExpSineSquared, ConstantKernel, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


def fit_gp(X, y, period):
    # long_term_trend_kernel = 50.0 ** 2 * RBF(length_scale=50.0)

    k0 = WhiteKernel()

    k1 = RBF(length_scale=int(period/10)) * ExpSineSquared(length_scale=1.0, periodicity=period)

    k2 = RBF(length_scale=int(period/10))

    kernel = k0 + k2 + k1

    # FINAL_KERNEL = RBF(length_scale=1) * ExpSineSquared(length_scale=1, periodicity=1) + RBF(
    #     length_scale=1) * RationalQuadratic(alpha=1, length_scale=1) * DotProduct(sigma_0=1) + RBF(
    #     length_scale=1) * RationalQuadratic(alpha=1, length_scale=1) + RationalQuadratic(alpha=1, length_scale=1)

    gp1 = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=0.0
    )

    # # Plot prior
    # gp1_prior_samples = gp1.sample_y(X=X, n_samples=100)
    # fig, ax = plt.subplots()
    # ax.plot(X[:, 1], y, color="black", linestyle="dashed", label="Observations")
    # ax.plot(X[:, 1], gp1_prior_samples, color="tab:blue", alpha=0.4, label="prior samples")
    # plt.legend()

    gp_fitted = gp1.fit(X, y)

    return gp_fitted, gp1

