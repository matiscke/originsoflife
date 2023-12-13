import paths
import pickle
import plotstyle

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
    )  # vector theta chosen at random for demonstrative purposes for simplicity we assume theta in 0,1

    nboot = 100000  # it is advised to use larger nboot because we have increased the sampling space quite a bit (probably already too much)
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
            label=showlabel + "True evidence for $H_\mathrm{3}$",
            c="C0",
            ls=ls,
        )
        ax.plot(
            gtvec,
            [np.sum(bfh0_kh0 > g) / len(bfh1_kh1) for g in gtvec],
            label=showlabel + "True evidence for $H_\mathrm{3, null}$",
            c="C1",
            ls=ls,
        )
    
    # ax.vlines(x=[1],ymin=0,ymax=1,linestyles='--',colors='grey')
    ax.vlines(x=[10], ymin=0, ymax=1, linestyles="-", lw=1, colors="gray")
    ax.text(
        9, 1, "strong", rotation=90, va="top", ha="right", c="k", alpha=1.0, fontsize=12
    )
    ax.vlines(x=[100], ymin=0, ymax=1, linestyles="-", lw=1, colors="gray")
    ax.text(
        94, 1, "extreme", rotation=90, va="top", ha="right", c="k", alpha=1.0, fontsize=12
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
    trueeH0 = np.sum(bfh0_kh0 > eta) / len(bfh0_kh0)
    return np.min([trueeH1, trueeH0])

def plot_evidence_sampsize(ax, s_max=500):
    """plot strong true evidence as a function of sample size"""
    svec = np.arange(10,s_max)
    true_strong_evidence_p = [P_true_evidence(sampsize=s) for s in svec]

    ax.plot(svec,true_strong_evidence_p)
    ax.set_ylim(-.05, 1.05)
    ax.set_ylabel('Probability of strong true evidence')
    ax.set_xlabel('Sample size')

    return ax


fig, axs = plt.subplots(1, 2, figsize=[13, 4.5])
axs[0] = plot_true_evidence(axs[0])
axs[1] = plot_evidence_sampsize(axs[1])
# axs[1] = plot_evidence_sampsize(axs[1], s_max=50)


# fig.tight_layout(
fig.savefig(paths.figures / "semian_true_evidence.pdf")