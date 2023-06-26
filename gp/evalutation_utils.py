from scipy.stats import norm


def calculate_ci(se, mean, alpha=0.05, dist=norm):
    return (mean - se * dist.ppf(1 - alpha / 2), mean + se * dist.ppf(1 - alpha / 2))



