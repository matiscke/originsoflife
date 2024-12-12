import pandas as pd

import paths
import dill
import plotstyle

plotstyle.styleplots()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

from bioverse.util import interpolate_luminosity, interpolate_nuv


NUV_thresh = 300.0


def get_example_planets(d):
    try:
        with open(paths.data / "pipeline/exampleplanets.dll", "rb") as file:
            print("Loading example planets from file.")
            sample = dill.load(file)

    except FileNotFoundError:
        dd = d.to_pandas()
        ids = dd[
            dd.planetID.isin(
                [
                    40278,
                    578,
                    71628,
                    # 3476,   8688,  15613,  40030,  54874,  56161,  71628,
                    # 79440, 106402, 121746, 149285, 170230, 174441, 180296,
                    # 185852, 189631, 208822, 221669, 224253
                ]
            )
        ].planetID
        sample = d[np.isin(d["planetID"], ids)]
        sample.evolution = {k: v for k, v in d.evolution.items() if k in ids.values}

        with open(paths.data / "pipeline/exampleplanets.dll", "wb") as file:
            dill.dump(sample, file)

    return sample


def plot_planets_scatter(ax, d, norm, col=None):
    # for (v, c) in [(False, 'white'), (True, 'C1')]:
    for c, (EEC, group) in zip(["white", "C0"], d.to_pandas().groupby("EEC")):
        if col != "white":
            col = group["L_st_interp"]
        ax.scatter(
            group["age"],
            group["M_st"],
            c=col,
            s=4,
            norm=norm,
            # edgecolors="white",
            edgecolors=c,
            linewidths=0.6,
            label=["EEC" if EEC else None][0],
        )
    ax.legend(loc="upper left", framealpha=0.0)
    return ax


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
        ax = plot_planets_scatter(ax, d, norm)
    fig.colorbar(q, label=cbarlabel)
    ax.set_xscale("log")
    ax.set_xlabel("Time (Gyr)")
    ax.set_ylabel("Stellar mass ($M_\odot$)")
    return ax


def plot_nuv_evo(fig, ax, log=True, d=None):
    interp_nuv = interpolate_nuv()
    T = np.geomspace(5e-3, 10.0, num=200)
    M = np.linspace(0.1, 1.0, num=200)
    M, T = np.meshgrid(M, T)
    Z = interp_nuv(M, T)
    plot_interpolation(
        fig, ax, T, M, Z, log=True, cbarlabel="NUV flux [erg/s/cm$^2$]", cmap="viridis"
    )

    # plot measured NUV fluxes
    nuv = pd.read_csv(paths.data / "past-UV.csv", comment="#")
    nuv["age"] /= 1000

    # translate spectral type to stellar mass using the mass midpoints in Richey-Yowell et al. (2023)
    spT2mass = {"K": 0.75, "earlyM": 0.475, "lateM": 0.215}
    nuv["Mst"] = nuv["SpT"].map(spT2mass)

    ax.scatter(
        nuv["age"],
        nuv["Mst"],
        s=25,
        marker="x",
        color="k",
        linewidths=0.6,
    )

    if d:
        if log:
            norm = colors.LogNorm()
        else:
            norm = colors.Normalize()
        ax = plot_planets_scatter(ax, d, norm, col="white")
    return ax


def plot_hz_and_nuv(fig, ax, sample, NUV_thresh=300.0, N_sample=2, random_state=44):
    "plot where HZ and sufficient NUV overlap for a few planets."
    offset = 0.01
    eec = sample[sample["EEC"].astype(bool)]
    # eec.evolve()
    eecdf = eec.to_pandas()
    inhabited = eecdf[eecdf.hz_and_uv]

    exampleplanets = {}
    # for i, id in enumerate(
    #     inhabited.sample(N_sample, random_state=random_state).planetID
    # ):
    for i, id in enumerate(inhabited.planetID):
        Mst = eecdf[eecdf.planetID == id]["M_st"]
        t = sample.evolution[id]["time"]
        # ax.plot(
        #     np.concatenate((np.array([1e-3]), t)),
        #     [Mst] + [Mst for tt in t],
        #     lw=1,
        #     c="gray",
        # )
        in_hz = sample.evolution[id]["in_hz"]
        nuv = sample.evolution[id]["nuv"]

        # check where planet was in the HZ and had NUV fluxes above the threshold value
        t_hz = t[in_hz]
        t_nuv = t[nuv > NUV_thresh]
        hz_and_uv = in_hz & (nuv > NUV_thresh)
        t_hzuv = t[hz_and_uv]

        ax.plot(t_hz, [Mst + offset for tt in t_hz], lw=3.5, c="xkcd:pale gold", label="HZ")
        ax.plot(t_nuv, [Mst - offset for tt in t_nuv], lw=3.5, c="C1", label="high NUV")

        # plot empty rectangle around overlapping region
        if len(t_hzuv) > 1:
            # h_off_left = offset*np.abs(np.log10(t_hzuv[0]))
            # h_off_right = offset*np.abs(np.log10(t_hzuv[-1]))

            ax.add_patch(
                Rectangle((t_hzuv[0] - 0.1*t_hzuv[0], Mst.values[0] - 4*offset),
                          t_hzuv[-1] - t_hzuv[0] + 0.2*t_hzuv[-1],  8*offset,
                fill=False, lw=3, edgecolor="white", zorder=99, label="HZ and high NUV")
            )


        # save Mst and last time to dictionary
        exampleplanets[Mst.values[0]] = t[-1]

        # print out the planet's SpT
        print(f"Planet {i+1} has SpT {eecdf[eecdf.planetID == id].SpT.values[0]}")

    # plot a not-inhabited planet, too
    # id = eecdf[~eecdf.hz_and_uv].sample(1, random_state=48).planetID.values[0]
    id = 40278
    Mst = eecdf[eecdf.planetID == id]["M_st"]
    t = sample.evolution[id]["time"]
    # ax.plot(
    #     np.concatenate((np.array([1e-3]), t)), [Mst] + [Mst for tt in t], lw=1, c="gray"
    # )

    t = sample.evolution[id]["time"]
    in_hz = sample.evolution[id]["in_hz"]
    t_hz = t[in_hz]
    ax.plot(t_hz, [Mst + offset for tt in t_hz], lw=3.5, c="xkcd:pale gold")

    exampleplanets[Mst.values[0]] = t.tolist()[-1]
    print(f"Planet {i+2} has SpT {eecdf[eecdf.planetID == id].SpT.values[0]}")

    # add number to the example planets, ordered decending by mass
    for i, (Mst, t) in enumerate(
        sorted(exampleplanets.items(), key=lambda x: x[0], reverse=True)
    ):
        ax.text(
            t,
            Mst,
            f"{i+1}",
            color="white",
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


    return ax


if __name__ == "__main__":
    # with open(paths.data / "pipeline/sample.dll", "rb") as f:
    # with open(paths.data / "pipeline/sample_FGK.dll", "rb") as f:
    with open(paths.data / "pipeline/sample_all.dll", "rb") as f:
        # with open(paths.data / "pipeline/sample_M.dll", "rb") as f:
        d = dill.load(f)

    m = np.linspace(0.1, 1.0, num=200)
    t = np.geomspace(1.4e-2, 10.0, num=200)
    M, T = np.meshgrid(m, t)
    interp_lum, extrap_nn = interpolate_luminosity()

    Z = interp_lum(M, T)
    d["L_st_interp"] = interp_lum(d["M_st"], d["age"])

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
    axs[0] = plot_interpolation(
        fig, axs[0], T, M, Z, d, cbarlabel="Bol. luminosity ($L/L_\odot$)"
    )
    axs[1] = plot_nuv_evo(fig, axs[1], d=d)

    # choose example planets
    sample = get_example_planets(d)

    axs[0] = plot_hz_and_nuv(fig, axs[0], sample, NUV_thresh=NUV_thresh)
    # add a legend covering only the elements 1, 2, and 3 in the axis:
    axs[0].legend(loc="lower left", framealpha=0.0, handles=axs[0].get_legend_handles_labels()[0][1:3],
                  bbox_to_anchor=(0.0, 1.0, 1.0, 0.1), ncol=3)


    axs[1] = plot_hz_and_nuv(fig, axs[1], sample, NUV_thresh=NUV_thresh)

    [ax.set_xlim(min(t), max(t)) for ax in axs]
    [ax.set_ylim(min(m), max(m)) for ax in axs]

    fig.tight_layout()
    fig.show()
    fig.savefig(paths.figures / "hz_nuv_evo.pdf")
