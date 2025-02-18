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
        palette={"EEC": "dimgray", "inhabited": "C1", "non-EEC": "darkgray"},
    )  # , hue_order = [True, False])
    # ax.set_yscale("log")

    return ax


def normalize_data(data):
    """Center, reduce, and scale data to [0, 1] range."""
    data = (data - data.mean()) / data.std()
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


def plot_nuv_distribution(sample, data, fig, ax, spt, plot_fit=False):
    sample = sample.to_pandas()
    dataa = data.to_pandas()
    normalized_data = normalize_data(dataa.max_nuv)

    # force-fit a beta distribution to our NUV_max
    # max_likeli = stats.beta.fit(dataa.max_nuv, method='MLE')
    max_likeli_norm = stats.beta.fit(normalized_data, method="MM")
    print(f"fit parameters: {max_likeli_norm}")

    # estimate selectivity from average of fitted beta function parameters
    selectivity = np.log10(1 / (np.mean(max_likeli_norm[:2])))
    save_var_latex("selectivity_{}".format(spt), round(selectivity, 2))
    print(f"selectivity ~ {selectivity:.2f}")

    # plot histogram and beta distribution fitted on non-normalized data
    x = np.arange(180.0, 610.0, 5)

    # define bins for histogram
    bins = np.linspace(180.0, 610.0, 28)

    max_likeli = stats.beta.fit(dataa.max_nuv, method="MM")
    # ax.hist(dataa.max_nuv, density=True, color="C1")
    ax.hist(
        [dataa.max_nuv[sample.inhabited], dataa.max_nuv[~sample.inhabited]],
        histtype="bar",
        stacked=True,
        density=True,
        color=["C1", "dimgray"],
        label=["inhabited", "EEC"],
        bins=bins,
    )

    if plot_fit:
        ax.plot(
            x,
            stats.beta.pdf(x, *max_likeli[:2], loc=max_likeli[2], scale=max_likeli[3]),
            # c="0.4",
            c="0.1",
        )

    ax.set_yticks([0, 0.01])
    ax.set_xlim([120, 610])
    ax.set_xlabel("max. $F_\mathrm{NUV}$ [erg/s/cm$^2$]")
    ax.set_ylabel("Probability density")
    # if spt == "M":
    #     ax.legend(title=None)

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
    eec["biosig"] = eec["biosig"].astype("bool")
    ax.scatter(
        eec[~eec.biosig]["max_nuv"], eec[~eec.biosig]["biosig"], s=9.0, color="dimgray"
    )
    ax.scatter(eec[eec.biosig]["max_nuv"], eec[eec.biosig]["biosig"], s=9.0, color="C1")
    ax.axvline(x=float(NUV_thresh), linestyle="--", color="grey")
    ax.set_yticks([0, 1])
    # ax.set_xlim([0,10])
    if ylabel:
        # ax.set_yticklabels(["$\oslash$", "$\checkmark$"], fontsize=16)
        ax.set_yticks([])
        ax.text(
            0.1,
            0.15,
            "no biosignature",
            fontsize=16,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            0.1,
            1.1,
            "biosignature",
            fontsize=16,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )
    else:
        ax.set_yticklabels(["", ""], fontsize=16)
        ax.set_yticks([])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # To turn off the bottom or left
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("max. $F_\mathrm{NUV}$ [erg/s/cm$^2$]")
    return fig, ax


def create_figure_grid(n_cases=2):
    """Create figure with grid for multiple deltaT cases."""
    fig = plt.figure(figsize=(10 * n_cases, 10))
    
    gs = gridspec.GridSpec(3, 2 * n_cases, 
                          height_ratios=[0.38, 0.38, 0.15], 
                          wspace=0.2, 
                          hspace=0.4)
    
    axes = []
    for case in range(n_cases):
        col_offset = case * 2
        # Create 3 rows x 2 columns of axes for each case
        case_axes = [
            fig.add_subplot(gs[0, col_offset]),   # First row, first column
            fig.add_subplot(gs[0, col_offset + 1]),  # First row, second column
            fig.add_subplot(gs[1, col_offset]),   # Second row, first column
            fig.add_subplot(gs[1, col_offset + 1]),  # Second row, second column
            fig.add_subplot(gs[2, col_offset]),   # Third row, first column
            fig.add_subplot(gs[2, col_offset + 1])   # Third row, second column
        ]
        
        # Share y axes within each case
        case_axes[1].sharey(case_axes[0])
        case_axes[3].sharey(case_axes[2])
        case_axes[5].sharey(case_axes[4])
        
        # Share x axes for bottom plots
        case_axes[4].sharex(case_axes[2])
        case_axes[5].sharex(case_axes[3])
        
        axes.append(case_axes)
    
    return fig, axes


def main():
    # Define the cases we want to plot
    cases = [
        {"suffix": "", "filename": "surveys_FGKM.pdf"},
        {"suffix": "_dT100", "filename": "surveys_FGKM_dT100.pdf"}
    ]
    
    for case in cases:
        # Create figure with grid for single case
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2, 
                             height_ratios=[0.38, 0.38, 0.15], 
                             wspace=0.2, 
                             hspace=0.4)
        
        # Create axes grid for this case
        axes = []
        for i in range(6):
            ax = fig.add_subplot(gs[i//2, i%2])
            axes.append(ax)
        
        # Share y axes
        axes[1].sharey(axes[0])
        axes[3].sharey(axes[2])
        axes[5].sharey(axes[4])
        
        # Share x axes for bottom plots
        axes[4].sharex(axes[2])
        axes[5].sharex(axes[3])
        
        axs_left = [axes[0], axes[2], axes[4]]
        axs_right = [axes[1], axes[3], axes[5]]
        
        for spt, axlr in zip(["FGK", "M"], [axs_left, axs_right]):
            # Load data with appropriate suffix
            with open(paths.data / f"pipeline/sample_{spt}{case['suffix']}.dll", "rb") as f:
                sample = dill.load(f)
            with open(paths.data / f"pipeline/data_{spt}{case['suffix']}.dll", "rb") as f:
                data = dill.load(f)

            # first row
            axlr[0] = plot_inhabited_FGKM(sample, fig, axlr[0])
            axlr[0].set_xlabel("Spectral Type")

            if spt == "M":
                ylabel = False
                leg = axlr[0].legend(
                    bbox_to_anchor=(1.07, 0.99),
                    title="N = {}".format(read_var_latex(f"N_nautilus{case['suffix']}")),
                    loc="lower right",
                    frameon=False,
                )
                leg._legend_box.align = "left"
            elif spt == "FGK":
                axlr[0].get_legend().remove()
                ylabel = True

            # second row
            fig, axlr[1] = plot_nuv_distribution(sample, data, fig, axlr[1], spt)

            # third row
            fig, axlr[2] = plot_detections_uv(
                data, fig, axlr[2], 
                NUV_thresh=read_var_latex(f"NUV_thresh{case['suffix']}"), 
                ylabel=ylabel
            )

            # Set titles for each column
            for ax in axlr:
                ax.set_title(spt, ha="right", x=0.95, va="top", y=0.9)

            # Add figure title for dT 100 case
            if case['suffix'] == '_dT100':
                fig.suptitle('$\Delta T_\mathrm{min}$ = 100 Myr', horizontalalignment='left', y=1.05)

            # Remove ylabel and tick labels from right plots
            if spt == "M":
                [ax.set_ylabel("") for ax in axlr]

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)  # Increase the height space between rows
        fig.savefig(paths.figures / case['filename'])
        plt.close()

if __name__ == "__main__":
    main()
