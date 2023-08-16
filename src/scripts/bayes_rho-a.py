import paths
import numpy as np
import matplotlib.pyplot as plt
import plotstyle
import astropy.units as u
from astropy.constants import G, M_sun
from scipy.special import expit, log_expit

plotstyle.styleplots()

rhoE = 3 / 4 * u.earthMass / (np.pi * u.earthRad**3)


def p_h1(rrho, aa, rho1=0.6, f_compress=20):
    """Environment factor for H1: above threshold density rho1"""
    # return np.array([[1 if (rho > rho1) else 0 for rho in rrho] for a in aa])
    return np.array([expit(f_compress * (rrho - rho1)) for a in aa])


def p_h2(rrho, aa, rho2=0.7, a2=0.01, f_compress=20):
    """Environment factor for H2: above threshold density rho1, sma a2"""
    # return np.array([[1 if ((rho > rho2) & (a > a2)) else 0 for rho in rrho] for a in aa])
    return np.array(
        [
            [
                expit(f_compress * (rho - rho2)) * expit(f_compress * (a - a2))
                for rho in rrho
            ]
            for a in aa
        ]
    )


def eval_Penv():
    """Evaluate environment factors on a grid in rho, a."""
    aa = np.geomspace(1e-3, 2.0, 30)
    rrho = np.linspace(0.2, 1.6, 30)
    P1 = p_h1(rrho, aa)
    P2 = p_h2(rrho, aa)
    return aa, rrho, P1, P2


def eval_Peta(aa, rrho):
    """Evaluate planet occurrence rate density on a grid in rho, a."""
    # SAG13 power law parameters
    R_break = 3.4
    gamma = [0.38, 0.73]
    alpha = [-0.19, -1.18]
    beta = [0.26, 0.59]

    def sma2period(a):
        """Convert semi-major axis to period."""
        return np.sqrt(4 * np.pi**2 * a * u.au**3 / (M_sun * G)).to(u.day).value

    def rho2radius(rho):
        """Convert bulk density to radius, assuming Earth density."""
        return np.cbrt(3 * u.earthMass / (4 * np.pi * rho * rhoE)).to(u.earthRad).value

    # Set up the probability grid in R and P
    lnx = np.log(sma2period(aa))
    lny = np.log(rho2radius(rrho))
    lnxv, lnyv = np.meshgrid(lnx, lny)
    dN = gamma[0] * (np.exp(lnxv) ** alpha[0]) * (np.exp(lnyv) ** beta[0])
    dN2 = gamma[1] * (np.exp(lnxv) ** alpha[1]) * (np.exp(lnyv) ** beta[1])
    dN[lnxv > np.log(R_break)] = dN2[lnxv > np.log(R_break)]
    return dN


def plot_factorgrid(aa, rrho, P, title=None, **kwargs):
    """Plot evaluated environment factors on a grid in rho, a."""
    RRHO, AA = np.meshgrid(rrho, aa)
    fig, ax = plt.subplots()
    ax.contourf(AA, RRHO, P, cmap="Blues", alpha=0.5)
    cs = ax.contour(AA, RRHO, P, cmap="Blues", **kwargs)
    h, _ = cs.legend_elements()
    ax.set_xscale("log")
    ax.set_xlabel("semi-major axis [au]")
    ax.set_ylabel(r"bulk density [$\rho_\oplus$]")
    ax.set_title(title)
    fig.colorbar(cs, label="$P_\mathrm{\eta}$")
    return fig, ax


def plot_factorgrid_dual(aa, rrho, P1, P2, title=None):
    """Plot evaluated environment factors on a grid in rho, a."""
    RRHO, AA = np.meshgrid(rrho, aa)
    fig, ax = plt.subplots()

    # dummy plot for colorbar
    dummy_cs = ax.pcolormesh(AA, RRHO, P2, cmap="gray_r", vmin=0.0, vmax=1.0)
    dummy_cs.set_visible(False)

    ax.contourf(AA, RRHO, P1, cmap="Blues", alpha=0.5, vmin=0.3)
    csf2 = ax.contourf(AA, RRHO, P2, cmap="Oranges", alpha=0.5, vmin=0.3)
    cs1 = ax.contour(AA, RRHO, P1, cmap="Blues", vmax=1.0)
    cs2 = ax.contour(AA, RRHO, P2, cmap="Oranges", vmax=1.0)
    h1, _ = cs1.legend_elements()
    h2, _ = cs2.legend_elements()
    ax.legend([h1[-1], h2[-1]], ["$P_\mathrm{env, 1}$", "$P_\mathrm{env, 2}$"])
    ax.set_xscale("log")
    ax.set_xlabel("semi-major axis [au]")
    ax.set_ylabel(r"bulk density [$\rho_\oplus$]")
    ax.set_title(title)
    fig.colorbar(dummy_cs, label="$P_\mathrm{env}$")
    return fig, ax


def eval_ln_bayesfactor(posterior1, posterior2):
    """evaluate the logarithm of the Bayes factor on a grid in rho, a."""
    return np.log(posterior1 / posterior2)


def plot_ln_bayesfactor(aa, rrho, bayesfactor):
    """Plot the Bayes factor on a grid in rho, a."""
    RRHO, AA = np.meshgrid(rrho, aa)
    fig, ax = plt.subplots()
    levels = np.array([1.0, 2.5, 5.0])
    ax.contourf(AA, RRHO, bayesfactor, alpha=0.5, levels=levels, cmap="Greens")
    cs = ax.contour(AA, RRHO, bayesfactor, levels=levels, cmap="Greens")
    h, _ = cs.legend_elements()
    # ax.legend([h[-1]], ["Bayes factor"])
    ax.set_xscale("log")
    ax.set_xlabel("semi-major axis [au]")
    ax.set_ylabel(r"bulk density [$\rho_\oplus$]")
    ax.set_title(r"Bayes Factor")
    fig.colorbar(cs, ticks=levels, label=r"$\ln(\mathcal{B})$")
    return fig, ax


if __name__ == "__main__":
    aa, rrho, P1, P2 = eval_Penv()
    fig, ax = plot_factorgrid_dual(aa, rrho, P1, P2)
    fig.savefig(paths.figures / "analytic/Penv.pdf")
    dN = eval_Peta(aa, rrho)
    fig, ax = plot_factorgrid(aa, rrho, dN, title=r"$P_\mathrm{\eta}$")
    fig.savefig(paths.figures / "analytic/Peta.pdf")
    posterior1 = P1 * dN
    posterior2 = P2 * dN
    fig, ax = plot_ln_bayesfactor(aa, rrho, eval_ln_bayesfactor(posterior1, posterior2))
    fig.savefig(paths.figures / "analytic/bayes_rho-a.pdf")
