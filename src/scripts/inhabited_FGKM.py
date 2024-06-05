import dill
import paths
import plotstyle
plotstyle.styleplots()

import seaborn as sns

def plot_inhabited_FGKM(d):
    """Plot the distribution of inhabited planets in the FGKM diagram.

    Parameters
    ----------
    d : bioverse Table
        The table containing the sample.

    Returns
    -------
    g : seaborn.FacetGrid
        a seaborn FacetGrid object containing the plot.
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
    
    g = sns.catplot(
        dd, x="SpT", kind="count", hue="Category", order=['F', 'G', 'K', 'M']
    )  # , hue_order = [True, False])
    g.ax.set_yscale("log")

    g.savefig(paths.figures / "inhabited_FGKM.pdf")
    return g


with open(paths.data / 'pipeline/sample.dll', 'rb') as f:
    sample = dill.load(f)

g = plot_inhabited_FGKM(sample)