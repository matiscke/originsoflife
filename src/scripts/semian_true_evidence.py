import paths
import dill
import plotstyle
import cmocean

plotstyle.styleplots()

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scp
from scipy import stats

rng = np.random.default_rng(seed=42)

def BFH1(k, theta, pi, theta_l):
    """
    Evidence for H1 against H0
    k number of life detection in the sample
    theta : vector of theta values in the sample
    pi : probability of life emergence
    theta_l : threshold of theta for life emergence

    returns bf the bayes ratio P(k|H1)/P(k|H0)
    """
    if len(np.shape(theta_l)) == 2:
        nl = np.sum(theta > theta_l, axis=1)
    elif len(np.shape(theta_l)) == 1:
        nl = np.sum(theta > np.reshape(theta_l, (len(theta_l), 1)), axis=1)
    else:
        nl = np.sum(theta > theta_l)
    bf = stats.binom.pmf(k, nl, pi) / stats.binom.pmf(k, len(theta), pi)
    return bf

def sample_H0(theta,pi,theta_l):
    """
    Generates a sample of values of k under H0
    """
    # theta is the vector of theta values
    # pi is the probability of life
    # theta_l is the value of the limit theta
    return(np.random.binomial(len(theta),pi))


def sample_H1(theta,pi,theta_l):
    """
    Generates a sample of values of k under H1
    """
    # theta is the vector of theta values
    # pi is the probability of life
    # theta_l is the value of the limit theta
    if len(np.shape(theta_l))==2:
        X = np.random.binomial(np.sum(theta>theta_l,axis=1),pi)
    elif len(np.shape(theta_l))==1:
        X = np.random.binomial(np.sum(theta>np.reshape(theta_l,(len(theta_l),1)),axis=1),pi)
    else:
        X = np.random.binomial(np.sum(theta>theta_l),pi)
    return(X)

def get_evidence(sampsize=100):
    # For convenience, we redefine theta here so that one can play with sample size and such
    theta = rng.uniform(
        0, 1, sampsize
    )  # vector theta chosen at random for demonstrative purposes; for simplicity we assume theta in 0,1

    nboot = 100000  # it is advised to use a large nboot because we have increased the sampling space quite a bit (probably already too much)
    pisamp = rng.uniform(0, 1, nboot)
    theta_lsamp = rng.uniform(0, 1, nboot)
    kh1samp = sample_H1(theta, pisamp, theta_lsamp)
    kh0samp = sample_H0(theta, pisamp, theta_lsamp)

    ## Calculate sample evidence
    bfh1_kh1 = BFH1(
        kh1samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H1 is true (k is sampled under H1)
    bfh1_kh0 = BFH1(
        kh0samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H0 is true (k is sampled under H0)
    bfh0_kh0 = 1 / bfh1_kh0  # evidence for H0 when H0 is true

    return bfh1_kh1, bfh0_kh0


def plot_true_evidence(ax):
    gtvec = 10 ** np.linspace(-2, 3, 300)

    for s, ls, showlabel in zip([10, 100], ["-", "--"], ["", "_"]):
        bfh1_kh1, bfh0_kh0 = get_evidence(s)
    
        ax.plot(
            gtvec,
            [np.sum(bfh1_kh1 > g) / len(bfh1_kh1) for g in gtvec],
            label=showlabel + "True evidence for $H_\mathrm{1}$",
            c="C0",
            ls=ls,
        )
        ax.plot(
            gtvec,
            [np.sum(bfh0_kh0 > g) / len(bfh1_kh1) for g in gtvec],
            label=showlabel + "True evidence for $H_\mathrm{null}$",
            c="C1",
            ls=ls,
        )

    # ax.vlines(x=[1],ymin=0,ymax=0.9,linestyles='--',colors='grey')
    ax.vlines(x=[10], ymin=0, ymax=1., linestyles="-", lw=1, colors="gray")
    ax.text(10, .95, " strong", va="bottom", ha="left", c="k", alpha=1.0, fontsize=12)
    ax.annotate(
        "",
        xy=(10, .93),
        xytext=(80, .93),
        arrowprops=dict(arrowstyle="<-", color="gray"),
    )
    ax.vlines(x=[100], ymin=0, ymax=0.9, linestyles="-", lw=1, colors="gray")
    ax.text(100, .85, " extreme", va="bottom", ha="left", c="k", alpha=1.0, fontsize=12)
    ax.annotate(
        "",
        xy=(100, .83),
        xytext=(1000, .83),
        arrowprops=dict(arrowstyle="<-", color="gray"),
    )
    ax.legend(
        fontsize=12,
        loc="lower left",
        ncol=99,
        bbox_to_anchor=(0.0, 0.99),
        frameon=False,
        columnspacing=1.6,
    )
    ax.set_xscale("log")
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel("P(evidence>x)")
    ax.set_xlabel("x")

    return ax

## calculate P(BF>eta)


def P_true_evidence(sampsize, eta=10, nboot=int(1e4), pisamp=None, theta_lsamp=None):
    theta = rng.uniform(0, 1, sampsize)

    if pisamp is None:
        pisamp = rng.uniform(0, 1, nboot)
    elif np.isscalar(pisamp):
        # try a fixed value for pi
        pisamp = np.full(nboot, pisamp)
    else:
        # seems we're dealing with an array
        pisamp = np.full((nboot, pisamp.shape[1]), pisamp)

    if theta_lsamp is None:
        # random threshold
        theta_lsamp = rng.uniform(0, 1, nboot)
    elif np.isscalar(theta_lsamp):
        # try a fixed threshold
        theta_lsamp = np.full(nboot, theta_lsamp)
    else:
        # seems we're dealing with an array
        theta_lsamp = np.full((nboot, theta_lsamp.shape[1]), theta_lsamp)

    kh1samp = sample_H1(theta, pisamp, theta_lsamp)
    kh0samp = sample_H0(theta, pisamp, theta_lsamp)

    bfh1_kh1 = BFH1(
        kh1samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H1 is true (k is sampled under H1)
    bfh1_kh0 = BFH1(
        kh0samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H0 is true (k is sampled under H0)
    bfh0_kh0 = 1 / bfh1_kh0

    trueeH1 = np.sum(bfh1_kh1 > eta) / len(bfh1_kh1)
    # trueeH0 = np.sum(bfh0_kh0 > eta) / len(bfh0_kh0)
    # return np.min([trueeH1, trueeH0])
    return trueeH1


def plot_evidence_sampsize(ax, s_max=500):
    """plot strong true evidence as a function of sample size"""
    svec = np.arange(10,s_max)
    true_strong_evidence_p = [P_true_evidence(sampsize=s) for s in svec]

    ax.plot(svec,true_strong_evidence_p)
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel('P($BF_{H_1, H_\mathrm{null}}$ > 10)')
    ax.set_xlabel('Sample size')

    return ax

def get_meshplot(smooth_sigma, sampsize, ax):
    x = np.linspace(0.0, 1.0, 10)  # theta_l
    y = np.linspace(0.0, 1.0, 10)  # pi
    z = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i, j] = P_true_evidence(
                sampsize=sampsize, eta=10, nboot=int(1e5), theta_lsamp=xi, pisamp=yj
            )

    # Smooth the data
    if smooth_sigma:
        from scipy.ndimage import gaussian_filter

        z = gaussian_filter(z, smooth_sigma)

    x, y = np.meshgrid(x, y, indexing="ij")
    im = ax.pcolormesh(
        x, y, z, cmap=cmocean.cm.ice
    )  # , vmin=vmin, vmax=vmax, cmap=cmap, lw=0, rasterized=True, shading='auto', edgecolors='k', linewidths=4)

    # ax.set_xlabel('$\\theta_{\lambda}$')
    ax.set_xlabel("NUV flux threshold (arb. units)")
    ax.set_ylabel("$f_\mathrm{life}$")

    return ax, im


def plot_evidence_grid():
    steps = 4
    sampsizes = np.rint(np.geomspace(10, 500, steps) / 10).astype(np.int64) * 10
    smooth_sigma = None

    fig, axs = plt.subplots(1, steps, figsize=[13, 3.3], sharey=True)

    for i, sampsize in enumerate(sampsizes):
        axs[i], im = get_meshplot(smooth_sigma, sampsize, axs[i])
        axs[i].text(0.99, 0.95, "$n = {}$".format(sampsize), horizontalalignment='right')

    cbar_ax = fig.add_axes([1.015, 0.18, 0.02, 0.775])
    cbar = fig.colorbar(im, label="P(true strong evidence)", cax=cbar_ax)
    fig.tight_layout()

    return fig, axs


def plot_beta(ax):
    # show Beta function for different parameters
    x = np.linspace(0, 1, 100)
    ax.plot(x, stats.beta.pdf(x, 0.1, 0.1), label=r"$s$=1, $Beta (0.1, 0.1)$")
    ax.plot(x, stats.beta.pdf(x, 1, 1), label=r"$s$=0, $Beta (1, 1)$")
    ax.plot(x, stats.beta.pdf(x, 10, 10), label=r"$s$=-1, $Beta (10,10)$")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density")
    ax.legend()
    return ax


def P_true_evidence_beta(
    sampsize, selectivity, eta=10, nboot=int(1e4), pisamp=None, theta_lsamp=None
):
    s = (
        10**selectivity
    )  # selectivity is log such that -1 is the center case, 0 is the neutral case and +1 is the border case
    theta = rng.beta(1 / s, 1 / s, sampsize)

    if pisamp is None:
        pisamp = rng.uniform(0, 1, nboot)
    elif np.isscalar(pisamp):
        # try a fixed value for pi
        pisamp = np.full(nboot, pisamp)
    else:
        # seems we're dealing with an array
        pisamp = np.full((nboot, pisamp.shape[1]), pisamp)

    if theta_lsamp is None:
        # random threshold
        theta_lsamp = rng.uniform(0, 1, nboot)
    elif np.isscalar(theta_lsamp):
        # try a fixed threshold
        theta_lsamp = np.full(nboot, theta_lsamp)
    else:
        # seems we're dealing with an array
        theta_lsamp = np.full((nboot, theta_lsamp.shape[1]), theta_lsamp)

    kh1samp = sample_H1(theta, pisamp, theta_lsamp)
    kh0samp = sample_H0(theta, pisamp, theta_lsamp)

    bfh1_kh1 = BFH1(
        kh1samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H1 is true (k is sampled under H1)
    bfh1_kh0 = BFH1(
        kh0samp, theta, pisamp, theta_lsamp
    )  # evidence for H1 when H0 is true (k is sampled under H0)
    bfh0_kh0 = 1 / bfh1_kh0

    trueeH1 = np.sum(bfh1_kh1 > eta) / len(bfh1_kh1)
    trueeH0 = np.sum(bfh0_kh0 > eta) / len(bfh0_kh0)
    return np.min([trueeH1, trueeH0])



def plot_selectivity(ax):
    X = np.arange(10, 300, 20)  # sample size
    Y = np.linspace(-1, 1, 10)  # selectivity
    Pvecz = np.vectorize(lambda x, y: P_true_evidence_beta(x, y, eta=10, nboot=int(1e4)))
    XX, YY = np.meshgrid(X, Y)
    Z = Pvecz(XX, YY)

    pc = ax.pcolormesh(XX, YY, Z, cmap=cmocean.cm.ice)
    fig.colorbar(pc, label="P(true strong evidence)")
    ax.set_ylabel("Selectivity $s$")
    ax.set_xlabel("Sample size $n$")
    ax.plot([10, 96], [0, 0], "--C1")
    ax.annotate(
        xytext=(20, 0.0),
        xy=(20, 0.3),
        text="",
        arrowprops=dict(arrowstyle="simple", facecolor="C1", connectionstyle="arc3"),
    )
    ax.annotate(
        xy=(25, 0.15), text=r"Sample more extreme $F_\mathrm{NUV}$", ha="left", va="center", color='k'
    )
    
    ax.annotate(
        xytext=(20, 0.0),
        xy=(20, -0.3),
        text="",
        arrowprops=dict(arrowstyle="simple", facecolor="C1", connectionstyle="arc3"),
    )
    ax.annotate(
        xy=(25, -0.15),
        text=r"Sample more intermediate $F_\mathrm{NUV}$",  color='k',
        ha="left",
        va="center",
    )
    return ax

def main():
    # 2-panel figure
    fig, axs = plt.subplots(1, 2, figsize=[13, 4.5])
    axs[0] = plot_true_evidence(axs[0])
    axs[1] = plot_evidence_sampsize(axs[1])
    # axs[1] = plot_evidence_sampsize(axs[1], s_max=50)

    # fig.tight_layout(
    fig.savefig(paths.figures / "semian_true_evidence.pdf")

    # evidence grid
    fig, axs = plot_evidence_grid()
    fig.savefig(paths.figures / "semian_evidence-grid.pdf")

    # selectivity figure
    fig, axs = plt.subplots(1, 2, figsize=[13, 4.5])
    axs[0] = plot_beta(axs[0])
    axs[1] = plot_selectivity(axs[1])

    fig.savefig(paths.figures / "semian_selectivity.pdf")

if __name__ == "__main__":
    main()