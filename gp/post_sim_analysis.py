from pathlib import Path
import numpy as np
from constants.constants import get_output_path
import pandas as pd
import matplotlib.pyplot as plt


split_dict = {"overall": ["overall_mean_covered", "covered_fraction_fun", "pred_logprob", "data_fraction",
                          "pred_prob_overall_mean", "meas_noise_var"],
         "train": ["covered_fraction_fun", "pred_logprob", "data_fraction", "log_marginal_likelihood", "meas_noise_var"],
         "test": ["covered_fraction_fun", "pred_logprob", "data_fraction", "meas_noise_var"]}


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


if __name__ == "__main__":
    output_path = Path("/home/gianna/Insync/OneDrive/master_thesis/repo_output/simulate_gp_616")
    perf_plot("overall", mode="ou_bounded_seasonal", file_path=output_path)
    perf_plot_split(file_path=output_path)

