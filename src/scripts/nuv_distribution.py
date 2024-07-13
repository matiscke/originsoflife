# Plot the distribution of measured maximum NUV irradiance for inhabited and non-inhabited planets in FGK and M stars, respectively.

import dill
import paths
import plotstyle
from src.scripts.utils import save_var_latex

plotstyle.styleplots()

import matplotlib.pyplot as plt

from scipy import stats
import numpy as np

def normalize_data(data):
    """Center, reduce, and scale data to [0, 1] range."""
    data = (data - data.mean()) / data.std()
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def plot_nuv_distribution(sample, data, fig, ax, spt):
    sample = sample.to_pandas()
    dataa = data.to_pandas()
    normalized_data = normalize_data(dataa.max_nuv)

    # force-fit a beta distribution to our NUV_max
    # max_likeli = stats.beta.fit(dataa.max_nuv, method='MLE')
    max_likeli_norm = stats.beta.fit(normalized_data, method="MM")
    print(f"fit parameters: {max_likeli_norm}")

    # estimate selectivity from average of fitted beta function parameters
    selectivity = np.log10(1 / (np.mean(max_likeli_norm[:2])))
    save_var_latex("selectivity_{}".format(spt), round(selectivity, 1))
    print(f"selectivity ~ {selectivity:.2f}")

    # plot histogram and beta distribution fitted on non-normalized data
    x = np.arange(0.0, 1000.0, 5)
    max_likeli = stats.beta.fit(dataa.max_nuv, method="MM")
    # ax.hist(dataa.max_nuv, density=True, color="C0")
    ax.hist([dataa.max_nuv[sample.inhabited], dataa.max_nuv[~sample.inhabited]], stacked=True, density=True, color=["C1", "C0"], label=["inhabited", "EEC"])
    ax.plot(
        x,
        stats.beta.pdf(x, *max_likeli[:2], loc=max_likeli[2], scale=max_likeli[3]),
        c="0.4",
    )

    ax.set_xlabel("max. NUV irradiance $F_\mathrm{NUV, max}$ [erg/s/$cm^2$]")
    ax.set_ylabel("Probability density")
    if spt == "FGK":
        ax.legend(title=None)


    # ax.text(
    #     0.97,
    #     0.85,
    #     transform=ax.transAxes,
    #     s="selectivity s = {:.1f}".format(selectivity),
    #     horizontalalignment="right",
    #     color="0.2",
    # )

    return fig, ax


fig, axs = plt.subplots(1, 2, figsize=(15, 2.5))

for spt, ax in zip(['FGK', 'M'], axs):
    with open(paths.data / "pipeline/sample_{}.dll".format(spt), "rb") as f:
        sample = dill.load(f)
    with open(paths.data / "pipeline/data_{}.dll".format(spt), "rb") as f:
        data = dill.load(f)

    fig, ax = plot_nuv_distribution(sample, data, fig, ax, spt)
    ax.set_title(f"{spt}-type host stars")

plt.show()
fig.savefig(paths.figures / "nuv_distribution.pdf")
