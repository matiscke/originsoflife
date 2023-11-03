import paths
import pickle
import plotstyle
plotstyle.styleplots()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from bioverse.util import interpolate_luminosity, interpolate_nuv

def plot_interpolation(fig, ax, T,M,Z, d=None, input_points=None, log=True, cbarlabel=None, **kwargs):
    if log:
        norm = colors.LogNorm()
    else:
        norm = colors.Normalize()
    q = ax.pcolormesh(T, M, Z, shading='auto',
                   norm=norm, rasterized=True, **kwargs)
    if input_points:
        ax.plot(*input_points, "x", ms=1, label="input point")
    if d:
        ax.scatter(d['age'], d['M_st'], c=d['L_st_interp'], s=4,
                norm=norm, edgecolors='white', linewidths=0.3, label="transiting planet")
    # ax.legend()
    fig.colorbar(q, label=cbarlabel)
    ax.set_xscale('log')
    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel('Stellar Mass ($M_\odot$)')
    return ax


def plot_nuv_evo(fig, ax):
    interp_nuv = interpolate_nuv()
    T = np.geomspace(0.01, 9., num=200)
    M = np.linspace(0.1, 0.9, num=200)
    M, T = np.meshgrid(M, T)
    Z = interp_nuv(M, T)
    plot_interpolation(fig, ax, T,M,Z, log=True, cbarlabel='NUV flux [erg/s/cm$^2$]', cmap='viridis')



with open(paths.data / 'pipeline/sample.pkl', 'rb') as f:
    d = pickle.load(f)

M = np.linspace(0.1, 1.4, num=200)
A = np.geomspace(5e-4, 10., num=200)
M, A = np.meshgrid(M, A)
interp_lum, extrap_nn = interpolate_luminosity()

Z = interp_lum(M, A)
d['L_st_interp'] = interp_lum(d['M_st'], d['age'])

fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
axs[0] = plot_interpolation(fig, axs[0], A,M,Z, d,  cbarlabel='Bol. luminosity ($L/L_\odot$)')
axs[1] = plot_nuv_evo(fig, axs[1])
fig.tight_layout()
fig.savefig(paths.figures / "hz_nuv_evo.pdf")
