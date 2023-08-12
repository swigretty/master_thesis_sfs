from pathlib import Path
import numpy as np
from constants.constants import get_output_path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

split_dict = {"overall": ["overall_mean_covered", "covered_fraction_fun", "pred_logprob", "data_fraction",
                          "pred_prob_overall_mean", "meas_noise_var"],
         "train": ["covered_fraction_fun", "pred_logprob", "data_fraction", "log_marginal_likelihood", "meas_noise_var"],
         "test": ["covered_fraction_fun", "pred_logprob", "data_fraction", "meas_noise_var"]}


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


def target_measure_perf_plot(target_measures_df, annotate="mse"):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(target_measures_df["method"].unique())))
    cdict = {i: color for i, color in enumerate(colors)}
    # cdict = {0: 'red', 1: 'blue', 2: 'green', 3: "orange", 4: "purple", 5: ""}

    method_col_map = {meth: i for i, meth in enumerate(target_measures_df["method"].unique())}
    target_measures_df["color"] = target_measures_df["method"].apply(lambda x: cdict[method_col_map[x]])
    fig, ax = plt.subplots(nrows=1, ncols=len(target_measures_df["data_fraction"].unique()), figsize=(14, 4))
    for i, (data_fraction, dff) in enumerate(target_measures_df.groupby("data_fraction")):
        ax[i].set_title(f"{data_fraction=}")
        for ii, (method, df) in enumerate(dff.groupby("method")):
            ax[i].scatter(df["ci_width"], df["ci_covered"], s=20, c=df["color"], marker='o', label=method)
            ax[i].set_ylim(- 0.1, 1.1)
            x_range = np.max(dff["ci_width"]) - np.min(dff["ci_width"])
            ax[i].set_xlim(np.min(dff["ci_width"]) - 0.2 * x_range, np.max(dff["ci_width"]) + 0.2 * x_range)

            if annotate:
                x_offset, y_offset = get_offset_annotate(ii, x_range)
                ax[i].annotate(f"{df[annotate].values[0]:.3f}", xy=(df["ci_width"], df["ci_covered"]),
                               xytext=(df["ci_width"] + x_offset, df["ci_covered"] + y_offset),
                               )
        # ax[i].plot(np.arange(np.min(dff["ci_width"]), np.max(dff["ci_width"])))
        ax[i].axhline(0.95, color="black", linestyle="dashed")

    ax[0].legend()
    ax[1].set_xlabel("ci width")
    ax[0].set_ylabel("ci coverage")
    return fig


def perf_plot(split="overall", mode=None, file_path=None):
    def is_numeric(col):
        try:
            pd.to_numeric(col)
            return True
        except Exception:
            return False

    if file_path is None:
        file_path = get_output_path()
    # col_of_int = split_dict[split]

    test_perf = pd.read_csv(file_path / f"{split}_perf.csv", index_col=False)
    if mode is not None:
        test_perf = test_perf[test_perf["mode"] == mode]
    col_of_int = [col for col in test_perf.columns if is_numeric(test_perf[col])]

    fig, ax = plt.subplots(ncols=len(col_of_int), nrows=len(col_of_int), figsize=(20, 20))
    pd.plotting.scatter_matrix(test_perf[col_of_int], alpha=0.8, ax=ax)
    fig.savefig(file_path / f"{split}_perf_scatter_matrix.pdf")


def perf_plot_split(data_fraction=0.1, file_path=None):

    if file_path is None:
        file_path = get_output_path()

    col_of_int = next(iter(split_dict.values()))
    for v in split_dict.values():
        col_of_int = np.intersect1d(col_of_int, v)

    dfs = {split: pd.read_csv(file_path / f"{split}_perf.csv")[col_of_int] for split in split_dict.keys()}
    for split, df in dfs.items():
        if split == "overall":
            sn = 1
        elif split == "train":
            sn = 0
        elif split == "test":
            sn = 2
        df["split"] = sn

    df_all = pd.concat(dfs.values())
    df_all = df_all[df_all["data_fraction"] == data_fraction]
    df_all.pop("data_fraction")

    fig, ax = plt.subplots(ncols=len(df_all.columns), nrows=len(df_all.columns), figsize=(10, 10))
    pd.plotting.scatter_matrix(df_all, alpha=0.8, ax=ax)
    fig.savefig(OUTPUT_PATH / f"split_perf_scatter_matrix_{data_fraction}.pdf")


MODES = ["sin_rbf_default", "sin_rbf_seasonal_default", "sin_rbf_seasonal_extreme"]


def plot_all(experiment_name, modes=MODES, annotate="mse"):
    output_path = Path(f"/home/gianna/Insync/OneDrive/master_thesis/repo_output/gp_experiments/{experiment_name}")
    target_measures_df = pd.read_csv(output_path / "target_measures_eval.csv")
    for mode in modes:
        for target_measure in target_measures_df["target_measure"].unique():
            df = target_measures_df[target_measures_df["output_path"].str.contains(mode) &
                                    (target_measures_df["target_measure"] == target_measure)]

            fig = target_measure_perf_plot(df.copy(), annotate=annotate)
            fig.savefig(output_path / f"{target_measure}_eval_{mode}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    experiment_name = "test_spline_reg2_n100"
    plot_all(experiment_name, annotate="mse")

    # output_path = Path("/home/gianna/Insync/OneDrive/master_thesis/repo_output/simulate_gp_616")
    # perf_plot("overall", mode="ou_bounded_seasonal", file_path=output_path)
    # perf_plot_split(file_path=output_path)

