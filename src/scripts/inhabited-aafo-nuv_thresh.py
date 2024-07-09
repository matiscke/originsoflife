"""
Plot the fraction of inhabited planets as a function of the NUV threshold
for samples of 100 planets orbiting FGK stars and M stars, respectively.
"""

import paths
import dill
import numpy as np
import matplotlib.pyplot as plt
from utils import timeit

from tqdm import tqdm

# from bioverse.generator import Generator
from bioverse_pipeline import (
    get_params_past_uv,
    generate_generator,
)

# Set a seed for reproducibility of the overall process
np.random.seed(42)


# Function to run a single simulation and return the fraction of inhabited planets
def run_simulation(nuv_thresh, n_planets, star_type, seed):
    np.random.seed(seed)

    params_past_uv = get_params_past_uv(star_type, NUV_thresh=nuv_thresh, seed=seed)
    g, g_args = generate_generator(label=None, **params_past_uv)
    d = g.generate()
    d = d.to_pandas()

    # sample n_planets, avoiding error when the sample is smaller than n_planets
    if len(d) < n_planets:
        d = d.sample(len(d), random_state=seed, replace=True)
    else:
        d = d.sample(n_planets, random_state=seed)

    inhabited_planets = len(d[d["inhabited"]])
    return inhabited_planets / n_planets


def simulate():
    # set_params
    n_planets = 100
    n_nuv_thresholds = 10
    # n_nuv_thresholds = 2
    n_simulations_per_threshold = 10
    # n_simulations_per_threshold = 2
    # nuv_thresholds = np.geomspace(30.0, 3000.0, n_nuv_thresholds)
    # nuv_thresholds = np.geomspace(200.0, 600.0, n_nuv_thresholds)
    nuv_thresholds = np.linspace(50.0, 500.0, n_nuv_thresholds)
    # f_life = 1.0
    f_life = .8

    # Storage for results
    results_fgk = np.zeros((n_nuv_thresholds, n_simulations_per_threshold))
    results_m = np.zeros((n_nuv_thresholds, n_simulations_per_threshold))

    # Running simulations for FGK stars
    print("Running simulations for FGK stars...")
    for i, nuv_thresh in enumerate(tqdm(nuv_thresholds, desc="FGK NUV Thresholds")):
        for j in tqdm(
            range(n_simulations_per_threshold),
            desc=f"Simulations for NUV {nuv_thresh:.2f}",
            leave=False,
        ):
            seed = np.random.randint(0, 1e6)
            results_fgk[i, j] = run_simulation(
                nuv_thresh, n_planets, star_type="FGK", seed=seed
            )

    # Running simulations for M stars
    print("Running simulations for M stars...")
    for i, nuv_thresh in enumerate(tqdm(nuv_thresholds, desc="M NUV Thresholds")):
        for j in tqdm(
            range(n_simulations_per_threshold),
            desc=f"Simulations for NUV {nuv_thresh:.2f}",
            leave=False,
        ):
            seed = np.random.randint(0, 1e6)
            results_m[i, j] = run_simulation(
                nuv_thresh, n_planets, star_type="M", seed=seed
            )

    # Save results to disk
    with open(paths.data / "frac-inhabited_fgk.dll", "wb") as f:
        dill.dump((results_fgk, nuv_thresholds), f)

    with open(paths.data / "frac-inhabited_m.dll", "wb") as f:
        dill.dump((results_m, nuv_thresholds), f)


def plot_results():
    with open(paths.data / "frac-inhabited_fgk.dll", "rb") as f:
        results_fgk, nuv_thresholds_fgk = dill.load(f)

    with open(paths.data / "frac-inhabited_m.dll", "rb") as f:
        results_m, nuv_thresholds_m = dill.load(f)

    # Calculate the mean fraction of inhabited planets for each NUV threshold
    mean_fractions_fgk = np.mean(results_fgk, axis=1)
    mean_fractions_m = np.mean(results_m, axis=1)

    # Calculate the confidence intervals
    lower_bound_fgk = np.percentile(results_fgk, 5, axis=1)
    upper_bound_fgk = np.percentile(results_fgk, 95, axis=1)
    lower_bound_m = np.percentile(results_m, 5, axis=1)
    upper_bound_m = np.percentile(results_m, 95, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot FGK stars
    ax.plot(nuv_thresholds_fgk, mean_fractions_fgk, label="FGK Stars", marker="o")
    ax.fill_between(nuv_thresholds_fgk, lower_bound_fgk, upper_bound_fgk, alpha=0.3)

    # Plot M stars
    ax.plot(nuv_thresholds_m, mean_fractions_m, label="M Stars", marker="s")
    ax.fill_between(nuv_thresholds_m, lower_bound_m, upper_bound_m, alpha=0.3)

    ax.set_xlabel("NUV Threshold")
    ax.set_ylabel("Fraction of Inhabited Planets")
    # ax.set_xscale("log")
    # ax.set_title("Fraction of Inhabited Planets vs. NUV Threshold")
    ax.legend()
    fig.show()
    fig.savefig(paths.figures / "inhabited_aafo_nuv_thresh.pdf")


@timeit
def main():
    simulate()
    plot_results()


if __name__ == "__main__":
    main()
