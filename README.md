
## Analysis of Irregularly Spaced Time Series: A Gaussian Process Approach


## Description

This is the code base of the Master's Thesis titled: "Analysis of Irregularly Spaced Time Series:
A Gaussian Process Approach".
Motivated by a real-world
example, this study highlights Gaussian processes as a potent tool for analyzing irregularly
sampled time series data.
Using a simulated blood pressure dataset designed to mimic real-world dynamics, includ-
ing cyclic, autoregressive, and long-term trend components, we evaluate Gaussian process
regression’s performance in estimating blood pressure values from one week of irregularly
spaced measurements. We assess the accuracy of credible interval estimation for clini-
cally relevant target measures through repeated simulations, comparing it with baseline
methods, such as spline and linear regression, accompanied by bootstrapped confidence
intervals.

## Installation
This Code has been run with Python 3.10.13. 
Install the packages specified in the requirements.txt

## Usage

### Run Simulation Experiment
To reproduce the simulation results, run 100 simulation experiment 
across different data fractions (downsampling factors) and different modes 
(uniform, seasonal and extreme seasonal sampling) just run
"gp/gp_experiments.py".
Or run the following:

    from gp.gp_experiments import evaluate_data_fraction_modes, MODES

    experiment_name: str = "my_experiment"
    data_fraction: tuple = (0.05, 0.1, 0.2, 0.4)

    for mode in MODES:
        evaluate_data_fraction(mode,
                               n_samples=100,
                               experiment_name=experiment_name,
                               normalize_kernel=False,
                               normalize_y=True,
                               data_fraction=data_fraction)


This will generate the results of the simulation experiment
and store this under experiment_path:

    from constants.constants import OUTPUT_PATH_BASE
    experiment_path = OUTPUT_PATH_BASE / f"{Path(__main__.__file__).stem}" / experiment_name

Please adapt OUTPUT_PATH_BASE in "constants/constants.py" accordingly.
If you used "gp/gp_experiments.py" to run this the experiment_path evaluates
to:

    experiment_path = OUTPUT_PATH_BASE / "gp_experiments" / experiment_name

The main output produced is "target_measures_eval.csv"
which stores the performance of the different regression methods
across different data densities and sampling patterns.

There will be one folder created per mode and per data fraction.
Note that every simulation run involves sampling from the true GP.
The true GP is specified by MODES, which is a list of 
"gp.simulate_gp_config.GPSimulatorConfig" instances.


###  Generate CI Coverage - CI Width Plots

To generate the CI coverage - CI width plots run "gp/post_sim_analysis.py"
or run: 

    from gp.post_sim_analysis import plot_all

    experiment_name = "my_experiment"
    plot_all(experiment_name, annotate=None, reextract=False)

This will read the "target_measures_eval.csv" created before and store
one plot per sampling pattern in the experiment_path.


###  Plot Regression Methods Prediction Examples

To draw 10 samples from the true GP, fit the regression methods and produce
plots of the predictions run:

    import numpy as np
    from gp.simulate_gp_config import GPSimulatorConfig
    
    experiment_name = "my_plots"
    # Use uniform sampling and call the session "sin_rbf_default"
    mode = GPSimulatorConfig(session_name="sin_rbf_default")
    # Set the seed of the rng for reproducibility
    rng = np.random.default_rng(18)
    plot_sample(normalize_kernel=False, rng=rng,
                experiment_name=experiment_name, nplots=10, config=mode,
                data_fraction=0.05, normalize_y=True)

