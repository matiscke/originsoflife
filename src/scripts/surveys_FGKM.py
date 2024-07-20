import dill
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
        hue="Category", order=['F', 'G', 'K', 'M'],
        palette={"EEC": "dimgray", "inhabited": "C0", "non-EEC": "darkgray"}

    )  # , hue_order = [True, False])
    # ax.set_yscale("log")

    return ax











fig = plt.figure(figsize=(10, 8))

gs = gridspec.GridSpec(3, 2, height_ratios=[0.38, 0.38, 0.24])

ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[0, 1])  # First row, second column
ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
ax3 = fig.add_subplot(gs[1, 1])  # Second row, second column
ax4 = fig.add_subplot(gs[2, 0])  # Third row, first column
ax5 = fig.add_subplot(gs[2, 1])  # Third row, second column

# axs = [ax0, ax1, ax2, ax3, ax4, ax5]
axs_left = [ax0, ax2, ax4]
axs_right = [ax1, ax3, ax5]

[ax.set_title('FGK-type hosts') for ax in axs_left]
[ax.set_title('M-type hosts') for ax in axs_right]


for spt, axlr in zip(["FGK", "M"], [axs_left, axs_right]):
    with open(paths.data / "pipeline/sample_{}.dll".format(spt), "rb") as f:
        sample = dill.load(f)

    # first row
    axlr[0] = plot_inhabited_FGKM(sample, fig, axlr[0])
    axlr[0].text(0.97, 0.9, f"N = {len(sample)}", transform=axlr[0].transAxes, horizontalalignment="right", color="0.2")

    if spt == "M":
        axlr[0].legend(loc='upper left', title=None)
        axlr[0].set_xlabel('')
    elif spt == "FGK":
        axlr[0].get_legend().remove()





plt.tight_layout()
# plt.subplots_adjust(hspace=0.5)  # Increase the height space between rows
fig.savefig(paths.figures / "surveys_FGKM.pdf")
