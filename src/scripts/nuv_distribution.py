import dill
import paths
import plotstyle
from src.scripts.utils import save_var_latex

plotstyle.styleplots()

import matplotlib.pyplot as plt

from scipy import stats
import numpy as np


def plot_nuv_distribution(data):
    dataa = data.to_pandas()
    fig, ax = plt.subplots(figsize=(15, 1.5))

    # force-fit a beta distribution to our NUV_max
    # max_likeli = stats.beta.fit(dataa.max_nuv, method='MLE')
    max_likeli = stats.beta.fit(dataa.max_nuv, method="MM")
    print(f"fit parameters: {max_likeli}")

    # estimate selectivity from average of fitted beta function parameters
    selectivity = np.log10(1 / (np.mean(max_likeli[:2])))
    save_var_latex("selectivity_transit_volume-lim", round(selectivity, 1))
    print(f"selectivity ~ {selectivity:.2f}")

    x = np.arange(0.0, 1000.0, 5)
    fig, ax = plt.subplots()
    ax.hist(dataa.max_nuv, density=True, color="C0")
    ax.plot(
        x,
        stats.beta.pdf(x, *max_likeli[:2], loc=max_likeli[2], scale=max_likeli[3]),
        c="0.4",
    )

    ax.set_xlabel("max. NUV irradiance $F_\mathrm{NUV, max}$ [erg/s/$cm^2$]")
    ax.set_ylabel("Probability density")
    ax.text(
        0.97,
        0.9,
        transform=ax.transAxes,
        s="selectivity s = {:.1f}".format(selectivity),
        horizontalalignment="right",
        color="0.2",
    )

    return fig, ax


with open(paths.data / "pipeline/data.dll", "rb") as f:
    data = dill.load(f)

fig, ax = plot_nuv_distribution(data)

fig.savefig(paths.figures / "nuv_distribution.pdf")
