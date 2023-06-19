from pathlib import Path

import numpy as np

from constants.constants import OUTPUT_PATH
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("/home/gianna/Insync/OneDrive/master_thesis/repo_output/simulate_gp_616")


split_dict = {"overall": ["overall_mean_covered","covered_fraction_fun", "kl_fun", "pred_logprob", "data_fraction", "pred_prob_overall_mean"],
         "train": ["covered_fraction_fun", "kl_fun", "pred_logprob", "data_fraction", "log_marginal_likelihood"],
         "test": ["covered_fraction_fun", "kl_fun", "pred_logprob", "data_fraction"]}


def perf_plot(split="overall", mode="ou_bounded"):
    col_of_int = split_dict[split]
    test_perf = pd.read_csv(OUTPUT_PATH / f"{split}_perf.csv")
    test_perf = test_perf[test_perf["mode"] == mode]
    fig, ax = plt.subplots(ncols=len(col_of_int), nrows=len(col_of_int), figsize=(10, 10))
    pd.plotting.scatter_matrix(test_perf[col_of_int], alpha=0.8, ax=ax)
    fig.savefig(OUTPUT_PATH / f"{split}_perf_scatter_matrix.pdf")


def perf_plot_split(data_fraction=0.1):
    col_of_int = next(iter(split_dict.values()))
    for v in split_dict.values():
        col_of_int = np.intersect1d(col_of_int, v)

    dfs = {split: pd.read_csv(OUTPUT_PATH / f"{split}_perf.csv")[col_of_int] for split in split_dict.keys()}
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
    perf_plot("overall", mode="ou_bounded_seasonal")

    perf_plot_split()

