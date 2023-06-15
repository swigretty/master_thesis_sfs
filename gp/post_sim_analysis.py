from pathlib import Path
from constants.constants import OUTPUT_PATH
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("/home/gianna/Insync/OneDrive/master_thesis/repo_output/simulate_gp_615")


if __name__ == "__main__":
    col_of_int = ["covered_fraction_fun", "kl_fun", "pred_logprob", "data_fraction", "pred_prob_overall_mean"]
    test_perf = pd.read_csv(OUTPUT_PATH / f"overall_perf.csv")
    test_perf = test_perf[test_perf["mode"] == "ou_bounded"]
    fig, ax = plt.subplots(ncols=len(col_of_int), nrows=len(col_of_int), figsize=(10, 10))
    pd.plotting.scatter_matrix(test_perf[col_of_int], alpha=0.8, ax=ax)
    fig.savefig(OUTPUT_PATH / f"overall_perf_scatter_matrix.pdf")


