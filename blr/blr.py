from simulation.simulate_bp import simulate_bp_simple, evolving_sinus_seasonal
from functools import partial
import pymc3 as pm
from pymc3 import HalfCauchy, Model, Normal, glm, plot_posterior_predictive_glm, sample
import matplotlib.pyplot as plt

def model_simple(x, y):
    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10, testval=1.0)
        intercept = Normal("Intercept", 0, sigma=20)
        x_coeff = Normal("x", 0, sigma=20)

        # Define likelihood
        likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        trace = sample(3000, return_inferencedata=True)



if __name__ == "__main__":
    seasonal_fun = partial(evolving_sinus_seasonal, seas_ampl=10)
    ts_true = simulate_bp_simple(seasonal_fun=seasonal_fun)
    fig = ts_true.plot()
    plt.show(block=True)
    fig.savefig("plot.svg")
