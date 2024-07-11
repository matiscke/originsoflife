"""
Plot the fraction of inhabited planets as a function of the NUV threshold
for samples of 100 planets orbiting FGK stars and M stars, respectively.
"""

import paths
import dill
import numpy as np
import matplotlib.pyplot as plt

import plotstyle
from utils import timeit

from tqdm import tqdm

# from bioverse.generator import Generator
from bioverse_pipeline import (
    get_params_past_uv,
    generate_generator,
)

plotstyle.styleplots()

# Set a seed for reproducibility of the overall process
np.random.seed(42)


# Function to run a single simulation and return the fraction of inhabited planets
def run_simulation(nuv_thresh, n_planets, star_type, seed, generator_kwargs):
    np.random.seed(seed)

    params_past_uv = get_params_past_uv(star_type, NUV_thresh=nuv_thresh, seed=seed)
    for key, value in generator_kwargs.items():
        params_past_uv[key] = value

    g, g_args = generate_generator(label=None, **params_past_uv)
    d = g.generate()
    d = d.to_pandas()

    # sample n_planets, avoiding error when the sample is smaller than n_planets
    if len(d) < n_planets:
        d = d.sample(len(d), random_state=seed, replace=True)
    else:
        d = d.sample(n_planets, random_state=seed, replace=True)

    fraction = len(d[d["inhabited"]]) / len(d)
    print(fraction)
    return fraction


def simulate(transiting=True):
    # set_params
    n_planets = 500   # maximum number of planets to sample
    # n_nuv_thresholds = 10
    n_nuv_thresholds = 7
    # n_simulations_per_threshold = 10
    n_simulations_per_threshold = 4

    # nuv_thresholds = np.geomspace(30.0, 3000.0, n_nuv_thresholds)
    # nuv_thresholds = np.geomspace(200.0, 600.0, n_nuv_thresholds)
    nuv_thresholds = np.linspace(10.0, 800.0, n_nuv_thresholds)
    # f_life = 1.0

    # DEBUG
    generator_kwargs = {
    # 'f_life' : 0.8,
    'f_life' : 1.0,
    'deltaT_min' : 1., # Myr. Smaller values seems so slightly enhance #inhabited planets in the FGK sample.

    # increase sample size
    'f_eta' : 15,
    'transit_mode' : transiting,
    }

    if not transiting:
        # we have plenty of non-transiting planets without scaling up the occurrence rates
        generator_kwargs['f_eta'] = 1.5
        generator_kwargs['d_max'] = 37


    # Storage for results
    results_fgk = np.zeros((n_nuv_thresholds, n_simulations_per_threshold))
    results_m = np.zeros((n_nuv_thresholds, n_simulations_per_threshold))

    # Running simulations for FGK stars
    print(f"Running simulations for {'transiting' if transiting else 'all'} FGK stars...")
    for i, nuv_thresh in enumerate(tqdm(nuv_thresholds, desc="FGK NUV Thresholds")):
        for j in tqdm(
            range(n_simulations_per_threshold),
            desc=f"Simulations for NUV {nuv_thresh:.2f}",
            leave=False,
        ):
            seed = np.random.randint(0, 1e6)
            results_fgk[i, j] = run_simulation(
                nuv_thresh, n_planets, star_type="FGK", seed=seed,
                generator_kwargs=generator_kwargs)

    # Running simulations for M stars
    print(f"Running simulations for {'transiting' if transiting else 'all'} M stars...")
    for i, nuv_thresh in enumerate(tqdm(nuv_thresholds, desc="M NUV Thresholds")):
        for j in tqdm(
            range(n_simulations_per_threshold),
            desc=f"Simulations for NUV {nuv_thresh:.2f}",
            leave=False,
        ):
            seed = np.random.randint(0, 1e6)
            results_m[i, j] = run_simulation(
                nuv_thresh, n_planets, star_type="M", seed=seed,
                generator_kwargs=generator_kwargs)

    # Save results to disk
    lbl = 'transiting' if transiting else 'all'
    with open(paths.data / f"frac-inhabited_fgk_{lbl}.dll", "wb") as f:
        dill.dump((results_fgk, nuv_thresholds), f)

    with open(paths.data / f"frac-inhabited_m_{lbl}.dll", "wb") as f:
        dill.dump((results_m, nuv_thresholds), f)


def plot_results():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plotkwargs = {
    'transiting' : {'marker' : 'o'},
    'all' : {'marker' : 's', 'linestyle' : '--', 'alpha' : 0.6, 'linewidth' : 1.5},
    }

    for transiting, lbl in zip([True, False], ['transiting', 'all']):
        with open(paths.data / f"frac-inhabited_fgk_{lbl}.dll", "rb") as f:
            results_fgk, nuv_thresholds_fgk = dill.load(f)

        with open(paths.data / f"frac-inhabited_m_{lbl}.dll", "rb") as f:
            results_m, nuv_thresholds_m = dill.load(f)

        # Calculate the mean fraction of inhabited planets for each NUV threshold
        mean_fractions_fgk = np.mean(results_fgk, axis=1)
        mean_fractions_m = np.mean(results_m, axis=1)

        # Calculate the confidence intervals
        lower_bound_fgk = np.percentile(results_fgk, 5, axis=1)
        upper_bound_fgk = np.percentile(results_fgk, 95, axis=1)
        lower_bound_m = np.percentile(results_m, 5, axis=1)
        upper_bound_m = np.percentile(results_m, 95, axis=1)


        # Plot FGK stars
        ax.plot(nuv_thresholds_fgk, mean_fractions_fgk, label=f"FGK host stars ({lbl})", color='C0', **plotkwargs[lbl])
        ax.fill_between(nuv_thresholds_fgk, lower_bound_fgk, upper_bound_fgk, color=f"{'C0' if transiting else 'white'}", alpha=0.2)

        # Plot M stars
        ax.plot(nuv_thresholds_m, mean_fractions_m, label=f"M  host stars ({lbl})", color='C1', **plotkwargs[lbl])
        ax.fill_between(nuv_thresholds_m, lower_bound_m, upper_bound_m, color=f"{'C1' if transiting else 'white'}", alpha=0.2)

    ax.set_xlabel("$F_\mathrm{NUV, min}$ [erg/s/$cm^2$]")
    ax.set_ylabel("Fraction of inhabited planets")
    # ax.set_xscale("log")
    # ax.set_title("Fraction of Inhabited Planets vs. NUV Threshold")
    ax.legend()
    fig.show()
    fig.savefig(paths.figures / "inhabited_aafo_nuv_thresh.pdf")


@timeit
def main():
    for transiting in [True, False]:
    # for transiting in [False]:
    # for transiting in [True]:
    #     simulate(transiting)
        pass
    # DEBUG
    plot_results()


if __name__ == "__main__":
    main()
