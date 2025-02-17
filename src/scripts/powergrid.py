import dill
import paths
import plotstyle
import matplotlib.pyplot as plt
import cmocean
from bioverse import plots
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


plotstyle.styleplots()


def plot_powergrid(grid, fig, ax, **kwargs):
    """Plot the power grid."""
    labels = ("$F_\mathrm{NUV, min}$ [erg/s/$cm^2$]", "$f_\mathrm{life}$")

    # check if the dictionary grid contains 'p' or 'dlnZ' values
    if "p" in grid:
        method = "p"
    elif grid["dlnZ"] is not None:
        method = "dlnZ"
    else:
        method = "dBIC"

    fig, ax = plots.plot_power_grid(
        grid,
        method=method,
        axes=("NUV_thresh", "f_life"),
        labels=labels,
        log=(True, True),
        show=False,
        fig=fig,
        ax=ax,
        zoom_factor=50,
        smooth_sigma=.0,
        levels=None,
        cmap=cmocean.cm.dense_r,
        **kwargs,
    )

    ax.set_xlim(50, 1000)
    ax.set_ylim(bottom=1e-2, top=1.0)
    return fig, ax


def create_powergrid_figure(suffix=""):
    """Create and save a power grid figure for a specific case."""
    fig = plt.figure(figsize=[13, 4.])
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.03, 1, 0.08], wspace=0.25)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    cax = fig.add_subplot(gs[0, 3])

    for i, (spt, ax) in enumerate(zip(["FGK", "M"], axs)):
        with open(paths.data / f"pipeline/grid_flife_nuv_{spt}{suffix}.dll", "rb") as f:
            grid = dill.load(f)

        fig, ax = plot_powergrid(grid, fig, ax, cbar=False)

        if i == 1:
            ax.set_ylabel("")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:0.0f}".format(x)))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
        ax.set_title(f"{spt}-type host stars")

    fig.colorbar(axs[0].collections[0], cax=cax, label="Statistical power (%)")

    # Add figure title for dT 100 case
    if suffix == "_dT100":
        fig.suptitle('Minimum time in HZ: 100 Myr', y=1.02)

    fig.savefig(paths.figures / f"powergrid{suffix}.pdf")
    plt.close()


def main():
    # Create figures for both cases
    create_powergrid_figure()  # default case (1 Myr)
    create_powergrid_figure("_dT100")  # 100 Myr case


if __name__ == "__main__":
    main()
