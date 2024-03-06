import dill
import paths
import plotstyle
import matplotlib.pyplot as plt
import cmocean
from bioverse import plots

plotstyle.styleplots()

def plot_powergrid(grid):
    """Plot the power grid."""
    labels = ('S_thresh', 'f_life')

    fig, ax = plt.subplots()
    fig, ax = plots.plot_power_grid(grid,
                                        # method='p',
                                        method='dlnZ',
                                        axes=('NUV_thresh', 'f_life'), labels=labels,
                                        log=(True, False), show=False, fig=fig, ax=ax,
                                        # zoom_factor=2, smooth_sigma=.5,
                                        zoom_factor=0, smooth_sigma=.2,
                                        levels=[50, 95], cmap=cmocean.cm.dense_r)
    return fig, ax


with open(paths.data / 'pipeline/grid_flife_nuv.dll', 'rb') as f:
    grid = dill.load(f)

fig, ax = plot_powergrid(grid)
fig.savefig(paths.figures / "powergrid.pdf")