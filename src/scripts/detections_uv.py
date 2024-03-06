import pickle
import paths
import plotstyle
plotstyle.styleplots()

import matplotlib.pyplot as plt

def plot_detections_uv(eec):
    fig, ax = plt.subplots(figsize=(15, 1.5))
    ax.scatter(eec['max_nuv'], eec['has_O2'], s=9.)
    ax.set_yticks([0, 1])
    # ax.set_xlim([0,10])
    ax.set_yticklabels(['no biosignature', 'biosignature'],fontsize=24)
    ax.set_xlabel('max. NUV irradiance [erg/s/$cm^2$]',fontsize=24)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # To turn off the bottom or left
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, ax


with open(paths.data / 'pipeline/sample.pkl', 'rb') as f:
    sample = pickle.load(f)

fig, ax = plot_detections_uv(sample)

fig.savefig(paths.figures / "detections_uv.pdf")