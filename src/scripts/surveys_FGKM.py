import dill
import paths
import plotstyle
import seaborn as sns
from src.scripts.utils import save_var_latex, read_var_latex
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
import numpy as np

plotstyle.styleplots()


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
        dd,
        ax=ax,
        x="SpT",
        # kind="count",
        hue="Category",
        order=["F", "G", "K", "M"],
        palette={"EEC": "dimgray", "inhabited": "C0", "non-EEC": "darkgray"},
    )  # , hue_order = [True, False])
    # ax.set_yscale("log")

    return ax


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

    # define bins for histogram
    bins = np.linspace(0, 1000, 22)

    max_likeli = stats.beta.fit(dataa.max_nuv, method="MM")
    # ax.hist(dataa.max_nuv, density=True, color="C0")
    ax.hist(
        [dataa.max_nuv[sample.inhabited], dataa.max_nuv[~sample.inhabited]],
        stacked=True,
        density=True,
        color=["C0", "dimgray"],
        label=["inhabited", "EEC"],
        bins=bins,
    )
    ax.plot(
        x,
        stats.beta.pdf(x, *max_likeli[:2], loc=max_likeli[2], scale=max_likeli[3]),
        # c="0.4",
        c="0.1",
    )

    ax.set_xlabel("max. NUV irradiance $F_\mathrm{NUV, max}$ [erg/s/$cm^2$]")
    ax.set_ylabel("Probability density")
    if spt == "M":
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


def plot_detections_uv(eec, fig, ax, NUV_thresh, ylabel=True):
    eec = eec.to_pandas()
    eec["has_O2"] = eec["has_O2"].astype('bool')
    ax.scatter(eec[~eec.has_O2]["max_nuv"], eec[~eec.has_O2]["has_O2"], s=9.0, color="dimgray")
    ax.scatter(eec[eec.has_O2]["max_nuv"], eec[eec.has_O2]["has_O2"], s=9.0, color="C1")
    ax.axvline(x=float(NUV_thresh), linestyle="--", color="grey")
    ax.set_yticks([0, 1])
    # ax.set_xlim([0,10])
    if ylabel:
        ax.set_yticklabels(["$\oslash$", "$\checkmark$"], fontsize=16)
    else:
        ax.set_yticklabels(["", ""], fontsize=16)
        ax.set_yticks([])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # To turn off the bottom or left
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("max. NUV irradiance [erg/s/$cm^2$]")
    return fig, ax


fig = plt.figure(figsize=(10, 8))

gs = gridspec.GridSpec(3, 2, height_ratios=[0.38, 0.38, 0.24])

ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)  # First row, second column
ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)  # Second row, second column
ax4 = fig.add_subplot(gs[2, 0])  # Third row, first column
ax5 = fig.add_subplot(gs[2, 1])  # Third row, second column

# axs = [ax0, ax1, ax2, ax3, ax4, ax5]
axs_left = [ax0, ax2, ax4]
axs_right = [ax1, ax3, ax5]

[ax.set_title("FGK-type hosts") for ax in axs_left]
[ax.set_title("M-type hosts") for ax in axs_right]


for spt, axlr in zip(["FGK", "M"], [axs_left, axs_right]):
    with open(paths.data / "pipeline/sample_{}.dll".format(spt), "rb") as f:
        sample = dill.load(f)
    with open(paths.data / "pipeline/data_{}.dll".format(spt), "rb") as f:
        data = dill.load(f)

    # first row
    axlr[0] = plot_inhabited_FGKM(sample, fig, axlr[0])
    axlr[0].text(
        0.03,
        0.88,
        f"N = {len(sample)}",
        transform=axlr[0].transAxes,
        horizontalalignment="left",
        color="0.2",
    )

    if spt == "M":
        ylabel = False
        axlr[0].legend(loc="center left", title=None)
        axlr[0].set_xlabel("")
    elif spt == "FGK":
        axlr[0].get_legend().remove()
        ylabel = True

    # second row
    fig, axlr[1] = plot_nuv_distribution(sample, data, fig, axlr[1], spt)

    # third row
    fig, axlr[2] = plot_detections_uv(
        data, fig, axlr[2], NUV_thresh=read_var_latex("NUV_thresh"), ylabel=ylabel
    )

# remove ylabel and tick labels from all right plots
[ax.set_ylabel("") for ax in axs_right]
[ax.set_yticklabels([]) for ax in axs_right]

plt.tight_layout()
# plt.subplots_adjust(hspace=0.5)  # Increase the height space between rows
fig.savefig(paths.figures / "surveys_FGKM.pdf")
