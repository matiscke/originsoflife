import paths
import pickle
import plotstyle

plotstyle.styleplots()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from bioverse.util import interpolate_luminosity, interpolate_nuv


def plot_interpolation(
    fig, ax, T, M, Z, d=None, input_points=None, log=True, cbarlabel=None, **kwargs
):
    if log:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()
    q = ax.pcolormesh(T, M, Z, shading="auto", norm=norm, rasterized=True, **kwargs)
    if input_points:
        ax.plot(*input_points, "x", ms=1, label="input point")
    if d:
        # for (v, c) in [(False, 'white'), (True, 'C1')]:
        for c, (EEC, group) in zip(['white', 'C0'], d.to_pandas().groupby('EEC')):
            ax.scatter(
                group["age"],
                group["M_st"],
                c=group["L_st_interp"],
                s=4,
                norm=norm,
                # edgecolors="white",
                edgecolors=c,
                linewidths=0.6,
                label=["EEC" if EEC else None][0],
            )
        ax.legend(loc='upper left')
    fig.colorbar(q, label=cbarlabel)
    ax.set_xscale("log")
    ax.set_xlabel("Time (Gyr)")
    ax.set_ylabel("Stellar Mass ($M_\odot$)")
    return ax


def plot_nuv_evo(fig, ax):
    interp_nuv = interpolate_nuv()
    T = np.geomspace(5e-3, 10., num=200)
    M = np.linspace(0.1, 1.0, num=200)
    M, T = np.meshgrid(M, T)
    Z = interp_nuv(M, T)
    plot_interpolation(
        fig, ax, T, M, Z, log=True, cbarlabel="NUV flux [erg/s/cm$^2$]", cmap="viridis"
    )
    return ax


def plot_hz_and_nuv(fig, ax, sample, NUV_thresh=100.0, N_sample=2, random_state=42):
    "plot where HZ and sufficient NUV overlap for a few planets."
    eec = sample[sample["EEC"].astype(bool)]
    eec.evolve()
    eecdf = eec.to_pandas()
    inhabited = eecdf[eecdf.hz_and_uv]

    for id in inhabited.sample(N_sample, random_state=random_state).planetID:
        Mst = eecdf[eecdf.planetID == id]["M_st"]
        t = eec.evolution[id]["time"]
        ax.plot(t, [Mst for tt in t], lw=1, c="gray")

        in_hz = eec.evolution[id]["in_hz"]
        nuv = eec.evolution[id]["nuv"]
        # check if planet ever was in the HZ and had NUV fluxes above the threshold value
        hz_and_uv = in_hz & (nuv > NUV_thresh)

        t_hzuv = t[hz_and_uv]
        ax.plot(t_hzuv, [Mst for tt in t_hzuv], lw=4, c="C0")

    # plot a not-inhabited planet, too
    id = eecdf[~eecdf.hz_and_uv].sample(1, random_state=48).planetID.values[0]
    Mst = eecdf[eecdf.planetID == id]["M_st"]
    t = eec.evolution[id]["time"]
    ax.plot(t, [Mst for tt in t], lw=1, c="gray")

    return ax

if __name__ == "__main__":
    with open(paths.data / "pipeline/sample.pkl", "rb") as f:
        d = pickle.load(f)

    m = np.linspace(0.1, 1.0, num=200)
    t = np.geomspace(5e-3, 10.0, num=200)
    M, T = np.meshgrid(m, t)
    interp_lum, extrap_nn = interpolate_luminosity()

    Z = interp_lum(M, T)
    d["L_st_interp"] = interp_lum(d["M_st"], d["age"])

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
    axs[0] = plot_interpolation(
        fig, axs[0], T, M, Z, d, cbarlabel="Bol. luminosity ($L/L_\odot$)"
    )
    axs[1] = plot_nuv_evo(fig, axs[1])
    axs[0] = plot_hz_and_nuv(fig, axs[0], d, NUV_thresh=100.0)
    axs[1] = plot_hz_and_nuv(fig, axs[1], d, NUV_thresh=100.0)

    [ax.set_xlim(min(t), max(t)) for ax in axs]
    [ax.set_ylim(min(m), max(m)) for ax in axs]

    fig.tight_layout()
    fig.savefig(paths.figures / "hz_nuv_evo.pdf")
