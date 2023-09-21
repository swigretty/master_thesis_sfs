"""
This modules produces the CI coverage CI width plots from the
simulation experiment results produced by gp_experiments.py
"""
from pathlib import Path
import numpy as np
from constants.constants import RESULTS_PATH
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from logging import getLogger
from log_setup import setup_logging
from gp.gp_simulator import GPSimulationEvaluator
from gp.gp_plotting_utils import CB_color_cycle
from gp.gp_experiments import MODES

logger = getLogger(__name__)
plt.style.use('tableau-colorblind10')


split_dict = {"overall": ["overall_mean_covered", "covered_fraction_fun",
                          "pred_logprob", "data_fraction",
                          "pred_prob_overall_mean", "meas_noise_var"],
         "train": ["covered_fraction_fun", "pred_logprob", "data_fraction",
                   "log_marginal_likelihood", "meas_noise_var"],
         "test": ["covered_fraction_fun", "pred_logprob", "data_fraction",
                  "meas_noise_var"]}

MODE_NAMES = [mode.session_name for mode in MODES]

col_of_int = ["method", "data_fraction", "ci_width", "ci_covered_prop",
              "ci_covered_lb", "ci_covered_ub"]


rename_method_map = {"naive_overall_mean": "overall_mean"}


def get_offset_annotate(ii, x_range, y_range=1.2):
    x_offset = 0
    y_offset = 0
    if ii == 1:
        y_offset = - 0.05 * y_range
    if ii == 2:
        x_offset = - 0.2 * x_range
    if ii == 3:
        y_offset = - 0.05 * y_range
    if ii == 4:
        y_offset = - 0.05 * y_range
    return x_offset, y_offset


def target_measure_perf_plot(target_measures_df, annotate="mse",
                             ci_width_max=30, reextract=False,
                             ci_coverage_col="ci_covered_prop_v2"):
    # target_measures_df = target_measures_df[col_of_int].drop_duplicates()

    target_measures_df.method = target_measures_df.method.apply(
        lambda x: rename_method_map[x] if x in rename_method_map.keys() else x)

    colors = CB_color_cycle
    cdict = {i: color for i, color in enumerate(colors)}
    cdict = {0: 'red', 1: 'blue', 2: 'green', 3: "orange", 4: "purple",
             5: "cyan"}

    method_col_map = {meth: i for i, meth in enumerate(
        target_measures_df["method"].unique())}

    fig, ax = plt.subplots(nrows=1, ncols=len(
        target_measures_df["data_fraction"].unique()), figsize=(12, 3.3))
    for i, (data_fraction, dff) in enumerate(target_measures_df.groupby(
            "data_fraction")):
        cur_ax = ax
        if len(target_measures_df["data_fraction"].unique()) > 1:
            cur_ax = ax[i]
        dwnsplf = 1/data_fraction
        if not dwnsplf % 1:
            dwnsplf = int(dwnsplf)
        cur_ax.set_title(f"downsampling factor: "
                         f"{dwnsplf}")

        if reextract:
            dff = GPSimulationEvaluator.summarize_eval_target_measures(
                Path(dff["output_path"].values[0]) / "eval_measure_all.csv")
            dff = dff[dff.target_measure == reextract]
            dff.method = dff.method.apply(
                lambda x: rename_method_map[
                    x] if x in rename_method_map.keys() else x)

        dff["color"] = dff["method"].apply(lambda x: cdict[method_col_map[x]])

        dff = dff[dff["ci_width"] <= ci_width_max]
        for ii, (method, df) in enumerate(dff.groupby("method")):
            cur_ax.scatter(df["ci_width"], df[ci_coverage_col], s=20,
                           c=df["color"], marker='o', label=method)
            y_err = (np.abs(df[["ci_covered_lb", "ci_covered_ub"]].values
                            - df[ci_coverage_col].values.reshape(-1, 1))).T

            try:
                cur_ax.errorbar(df["ci_width"], df[ci_coverage_col],
                                yerr=y_err,
                                markerfacecolor=df["color"],
                                markeredgecolor=df["color"],
                                ecolor=df["color"])
            except Exception as e:
                print(method)
            cur_ax.plot(df["ci_width"], df["ci_covered_ub"])
            cur_ax.set_ylim(- 0.1, 1.1)
            x_range = np.max(dff["ci_width"]) - np.min(dff["ci_width"])
            cur_ax.set_xlim(np.min(dff["ci_width"]) - 0.2 * x_range,
                            np.max(dff["ci_width"]) + 0.2 * x_range)

            if annotate:
                x_offset, y_offset = get_offset_annotate(ii, x_range)
                cur_ax.annotate(
                    f"{df[annotate].values[0]:.3f}",
                    xy=(df["ci_width"], df[ci_coverage_col]),
                    xytext=(df["ci_width"] + x_offset,
                            df[ci_coverage_col] + y_offset),
                               )
        cur_ax.axhline(0.95, color="black", linestyle="dashed")
    cur_ax.legend()

    if len(target_measures_df["data_fraction"].unique()) > 1:
        # ax[1].set_xlabel("CI Width [mmHg]")
        ax[0].set_ylabel("CI Coverage []")
    return fig


def read_experiment(experiment_name, filter_dict=None,
                    table_name="target_measures_eval.csv",
                    results_path=RESULTS_PATH):
    output_path = results_path / experiment_name
    target_measures_df = pd.read_csv(output_path / table_name)
    if filter_dict:
        for k, v in filter_dict.items():
            if callable(v):
                mask = v(target_measures_df[k])
            else:
                mask = target_measures_df[k] == v
            target_measures_df = target_measures_df[mask]
    return target_measures_df, output_path


def plot_all(experiment_name, modes=MODE_NAMES, annotate="mse",
             filter_dict=None, reextract=False,
             ci_coverage_col="ci_covered_prop_v2", results_path=RESULTS_PATH):

    target_measures_df, output_path = read_experiment(
        experiment_name, filter_dict=filter_dict, results_path=results_path)

    for mode in modes:
        for target_measure in target_measures_df["target_measure"].unique():
            logger.info(f"Plot {mode=} and {target_measure=}")
            df = target_measures_df[
                target_measures_df["output_path"].str.contains(mode) &
                (target_measures_df["target_measure"] == target_measure)]
            if len(df) == 0:
                continue
            if reextract:
                reextract = target_measure
            fig = target_measure_perf_plot(df.copy(), annotate=annotate,
                                           reextract=reextract,
                                           ci_coverage_col=ci_coverage_col)
            xlabel = 'CI Width [mmHg]'
            if target_measure == "ttr":
                xlabel = 'CI Width []'
            fig.text(0.5, 0.02, xlabel, ha='center', va='center')
            fig.tight_layout()

            fig.savefig(output_path / f"{target_measure}_eval_{mode}.pdf")
            plt.close(fig)


def recreate_table_renamed(table_name):
    df, output_path = read_experiment(
        experiment_name, table_name=table_name)

    mode_pattern_map = {"sin_rbf_default": "uniform",
                        "sin_rbf_seasonal_default": "seasonal",
                        "sin_rbf_seasonal_extreme": "seasonal_extreme"}

    def get_sampling_pattern_from_output_path(str_path):
        for mode, pattern in mode_pattern_map.items():
            if mode in str_path:
                return pattern

    df["sampling_pattern"] = df["output_path"].apply(
        lambda x: get_sampling_pattern_from_output_path(x))
    df.to_csv(output_path / table_name)


if __name__ == "__main__":
    setup_logging()
    experiment_name = "my_experiment"
    plot_all(experiment_name, annotate=None, reextract=False,
             filter_dict={"method": lambda x: x != "gp_hdi"})
