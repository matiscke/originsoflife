import dill
import paths
import plotstyle

plotstyle.styleplots()

import matplotlib.pyplot as plt


def plot_detections_uv(eec, fig, ax):
    ax.scatter(eec["max_nuv"], eec["has_O2"], s=9.0)
    ax.set_yticks([0, 1])
    # ax.set_xlim([0,10])
    ax.set_yticklabels(["no biosignature", "biosignature"])#, fontsize=24)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # To turn off the bottom or left
    # ax.spines['bottom'].set_visible(False)
    ax.spines["left"].set_visible(False)
    return fig, ax


fig, axs = plt.subplots(2, 1, figsize=(15, 4.), sharex=True)
plt.subplots_adjust(hspace = 0.9)  # Adjust to the desired value
for spt, ax in zip(["FGK", "M"], axs):
    with open(paths.data / "pipeline/data_{}.dll".format(spt), "rb") as f:
        data = dill.load(f)

    fig, ax = plot_detections_uv(data, fig, ax)
    ax.set_title(f"{spt}-type host stars")

ax.set_xlabel("max. NUV irradiance [erg/s/$cm^2$]")#, fontsize=24)

fig.savefig(paths.figures / "detections_uv.pdf")
