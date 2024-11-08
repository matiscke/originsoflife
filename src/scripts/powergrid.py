import dill
import paths
import plotstyle
import matplotlib.pyplot as plt
import cmocean
from bioverse import plots
import matplotlib.ticker as ticker


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
        # log=(True, False),
        # log=(False, False),
        log=(True, True),
        show=False,
        fig=fig,
        ax=ax,
        zoom_factor=2,
        # zoom_factor=0,
        # smooth_sigma=.5,
        smooth_sigma=0.2,
        # levels=[50, 95],
        levels=None,
        cmap=cmocean.cm.dense_r,
        **kwargs,
    )

    ax.set_ylim(bottom=1e-3)
    return fig, ax


fig, axs = plt.subplots(1, 2, figsize=[13, 4.5])

for i, (spt, ax) in enumerate(zip(["FGK", "M"], axs)):
    with open(
        paths.data / "pipeline/grid_flife_nuv_{}.dll".format(spt), "rb"
    ) as f, open(paths.data / "pipeline/data_{}.dll".format(spt), "rb") as d:
        grid = dill.load(f)
        data = dill.load(d)  # do we need this?

    cbar = False if i == 0 else True

    fig, ax = plot_powergrid(grid, fig, ax, cbar=cbar)

    if i == 1:
        ax.set_ylabel("")

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:0.0f}".format(x))
    )
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    ax.set_title(f"{spt}-type host stars")


fig.savefig(paths.figures / "powergrid.pdf")
