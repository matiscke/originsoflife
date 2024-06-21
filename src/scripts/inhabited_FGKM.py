import dill
from matplotlib import pyplot as plt

import paths
import plotstyle
plotstyle.styleplots()

import seaborn as sns

def plot_inhabited_FGKM(d, fig, ax):
    """Plot the distribution of inhabited planets in the FGKM diagram.

    Parameters
    ----------
    d : bioverse Table
        The table containing the sample.
    fig : matplotlib figure
        The figure to plot on
    ax : matplotlib axis
        The axis to plot on

    Returns
    -------
    ax : matplotlib axis
        The axis containing the plot
    """

    dd = d.to_pandas()

    def cat(obj):
        if obj.EEC:
            cat = "EEC"
            if obj.inhabited:
                cat = "inhabited"
        else:
            cat = "non-EEC"
        return cat

    dd.loc[:, "Category"] = dd.apply(cat, axis=1)
    
    # g = sns.catplot(
    ax = sns.countplot(
        dd, ax=ax, x="SpT",
        # kind="count",
        hue="Category", order=['F', 'G', 'K', 'M']
    )  # , hue_order = [True, False])
    # ax.set_yscale("log")

    return ax


fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=False, sharey=True, gridspec_kw={"hspace": 0.3})
for spt, ax in zip(["FGK", "M"], axs):
    with open(paths.data / "pipeline/sample_{}.dll".format(spt), "rb") as f:
        sample = dill.load(f)

    ax = plot_inhabited_FGKM(sample, fig, ax)
    ax.set_title(f"{spt}-type host stars")

    # add sample size to the upper right corner
    ax.text(0.97, 0.9, f"N = {len(sample)}", transform=ax.transAxes, horizontalalignment="right", color="0.2")

    if ax is axs[0]:
        ax.legend(loc='upper left', title=None)
        ax.set_xlabel('')
    elif ax is axs[1]:
        ax.get_legend().remove()


fig.savefig(paths.figures / "inhabited_FGKM.pdf")
