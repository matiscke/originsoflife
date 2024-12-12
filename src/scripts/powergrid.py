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
        # log=(True, False),
        # log=(False, False),
        log=(True, True),
        show=False,
        fig=fig,
        ax=ax,
        zoom_factor=50,
        # zoom_factor=0,
        smooth_sigma=.0,
        # smooth_sigma=.5,
        # levels=[50, 95],
        levels=None,
        cmap=cmocean.cm.dense_r,
        **kwargs,
    )

    # ax.set_xlim(10, 1100)
    ax.set_xlim(50, 1000)
    # ax.set_ylim(bottom=1e-3, top=1.0)
    ax.set_ylim(bottom=1e-2, top=1.0)
    return fig, ax


fig = plt.figure(figsize=[13, 4.])
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.03, 1, 0.08], wspace=0.25)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
cax = fig.add_subplot(gs[0, 3])


for i, (spt, ax) in enumerate(zip(["FGK", "M"], axs)):
    with open(paths.data / "pipeline/grid_flife_nuv_{}.dll".format(spt), "rb") as f:
        grid = dill.load(f)

    fig, ax = plot_powergrid(grid, fig, ax, cbar=False)

    if i == 1:
        ax.set_ylabel("")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:0.0f}".format(x)))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    ax.set_title(f"{spt}-type host stars")

fig.colorbar(axs[0].collections[0], cax=cax, label="Statistical power (%)")

fig.show()
fig.savefig(paths.figures / "powergrid.pdf")
