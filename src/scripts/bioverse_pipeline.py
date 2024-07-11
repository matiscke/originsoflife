"""
This file contains the code to generate the datasets for the paper.
It heavily relies on Bioverse and its auxiliary functions.
"""
# import pickle
import dill

import paths
from utils import *

# import pickle
import numpy as np

# Import Bioverse objects
from bioverse.generator import Generator
from bioverse.survey import TransitSurvey
from bioverse.hypothesis import Hypothesis

# from bioverse.constants import CONST, DATA_DIR

from semian_true_evidence import BFH1


def eec_filter(d):
    """Filter the planets to only include EECs."""
    return d[d["EEC"].astype(bool)]


def inject_nuv_life_correlation(d, f_life=0.1):
    """Inject a correlation between past NUV flux and the occurrence of life.

    Uses the 'hz_and_uv' column created with Bioverse's `Table.evolve()` method that
    indicates whether a planet fulfills the following criteria:
     1. it is in the habitable zone now
     2. In its past, it simultaneously received a NUV flux above a threshold and was
     in the habitable zone for a minimum amount of time.

    Parameters
    ----------
    d : Table
        The table of planets to inject the correlation into.
    f_life : float
        The fraction of planets fulfilling the above criteria
        that are inhabited.

    Returns
    -------
    d : Table
        The table of planets with the injected correlation
        and including a new column `inhabited`.
    """

    d["inhabited"] = np.logical_and(
        d["hz_and_uv"], np.random.random(size=len(d)) <= f_life
    )
    return d


def inject_biosignature(d, P=1.0):
    # Randomly assign O2 to some inhabited planets
    d["has_O2"] = np.logical_and(d["inhabited"], np.random.random(size=len(d)) <= P)

    return d


def get_generator_args():
    """define default generator parameters. These may be overwritten by other functions"""
    # Parameters for star generation
    stars_args = {
        "d_max": 30,  # max. dist to stars (pc)
        "M_st_max": 1.5,  # Maximum stellar mass to consider, in solar units.
        # 'M_G_max' : 11,           # Maximum gaia magnitude of stars
        "M_G_max": 16,  # Maximum gaia magnitude of stars
        "seed": 42,  # seed for random number generators
        "lum_evo": True,  # luminosities from luminosity tracks (Baraffe et al. 1998), based on random ages
    }

    # Parameters for planet generation
    planets_args = {
        # "transit_mode": False,  # Simulate only transiting planets
        "transit_mode": True,  # Simulate only transiting planets
        "f_eta": 1.0,  # Occurrence rate scaling factor
        "R_min": 0.75,  # minimum radius for Bergsten et al. planet generator
        "P_max": 500.0,  # maximum period for Bergsten et al. planet generator
        "mr_relation": "Zeng2016",  # choice of mass-radius relationship ('Zeng2016'/'Wolfgang2016/'earthlike')
        # 'mr_relation' : 'Wolfgang2016',
        # 'mr_relation' : 'earthlike',
        # past HZ occupancy and NUV flux
        "deltaT_min": 100.0,  # Myr
        "NUV_thresh": 100.0,  # erg/s/cm2
    }
    return stars_args, planets_args


def generate_generator(stars_only=False, **kwargs):
    """Generate a Generator object with the required functional steps."""
    stars_args, planets_args = get_generator_args()
    g_args = stars_args | planets_args
    for key, value in kwargs.items():
        g_args[key] = value
    g = Generator(label=None)
    g.insert_step("read_stars_Gaia")
    if not stars_only:
        g.insert_step("create_planets_bergsten")
        g.insert_step("assign_orbital_elements")
        g.insert_step("geometric_albedo")
        g.insert_step("impact_parameter")
        g.insert_step("assign_mass")
        g.insert_step("effective_values")
        g.insert_step("compute_habitable_zone_boundaries")
        g.insert_step("classify_planets")
        g.insert_step("scale_height")
        g.insert_step("compute_transit_params")
        g.insert_step("past_hz_uv")
        # g.insert_step('apply_bias')
        g.insert_step(eec_filter)
        g.insert_step(inject_nuv_life_correlation)
        g.insert_step(inject_biosignature)
    [g.set_arg(key, val) for key, val in g_args.items()]
    return g, g_args


def create_survey_nautilus():
    """Create a survey object based on the Nautilus concept."""
    nautilus = TransitSurvey()

    # plan measurements and their precision
    margs = {}
    mkeys = [
        "L_st",
        "R_st",
        "M_st",
        "subSpT",
        "T_eff_st",
        "age",
        "depth",
        "R",
        "T_dur",
        "P",
        "a",
        "M",  # needed for estimates of HZ occupancy
        "S",
        "EEC",
        "has_O2",
        "hz_and_uv",
        "max_nuv",
    ]

    margs["precision"] = {
        "T_eff_st": 50.0,
        "R_st": "1%",
        "depth": "1%",
        "R": "5%",
        "M_st": "3%",
        "M": "5%",
        "age": "30%",
        "P": 0.000001,
        "S": "5%",
        # "max_nuv": "20%",
        "max_nuv": "5%",
    }

    # margs['t_ref'] = {'R':1.5}

    # Add the measurements to the survey
    for mkey in mkeys:
        mkwargs = {}
        for key, vals in margs.items():
            if mkey in vals:
                mkwargs[key] = vals[mkey]
        nautilus.add_measurement(mkey, **mkwargs)

    return nautilus


def run_survey_nautilus(sample, t_total=10 * 365.25):
    """Conduct a transit survey based on the Nautilus concept.

    Parameters
    ----------
    sample : Table
        The sample of planets to survey.
    t_total : float
        The total duration of the survey, in days.

    Returns
    -------
    sample : Table
        The sample of planets to survey.
    detected : Table
        The planets detected by the survey.
    data : Table
        A table with the obtained measurements and their errors.
    nautilus : TransitSurvey object
        The survey object.
    """
    nautilus = create_survey_nautilus()

    # compute yield, conduct survey
    detected = nautilus.compute_yield(sample)
    save_var_latex("N_nautilus", len(detected))
    data = nautilus.observe(
        detected
    )  # commented out for now because of Bioverse's issue #45: , t_total=t_total)

    # print(data['max_nuv'][:10])

    return sample, detected, data, nautilus


# hypothesis tests
def h1(theta, X):
    f_life, NUV_thresh = theta
    return f_life * (X >= NUV_thresh)


def h_null(theta, X):
    shape = (np.shape(X)[0], 1)
    return np.full(shape, theta)


# def hypothesis_test(data, method="dynesty"):
def hypothesis_test(data, testmethod):
    """Perform a single hypothesis test on the data."""

    # if testmethod == "binomial":
    #     bf = BFH1(k, theta, pi, theta_l)
    #
    #
    #     print("The evidence in favor of the hypothesis is: dlnZ = {:.1f} (corresponds to p = {:.1E})".format()
    #     return results
    #

    params = ("f_life", "NUV_thresh")
    log = (True, True)
    bounds = np.array([[1e-3, 1.0], [10.0, 1e5]])
    bounds_null = np.array([[1e-3, 1.0]])
    features = ("max_nuv",)
    labels = ("has_O2",)

    h_nuv = Hypothesis(
        h1, bounds, params=params, features=features, labels=labels, log=log
    )
    h_nuv.h_null = Hypothesis(
        h_null,
        bounds_null,
        params=("f_O2",),
        features=features,
        labels=labels,
        log=(True,),
    )

    results = h_nuv.fit(data, method=testmethod)

    if testmethod == "dynesty":
        print(
            "The evidence in favor of the hypothesis is: dlnZ = {:.1f} (corresponds to p = {:.1E})".format(
                results["dlnZ"], np.exp(-results["dlnZ"])
            )
        )
    elif testmethod == "emcee":
        print(
            "The differential Bayesian Information Criterion is dBIC={}".format(
                results["dBIC"]
            )
        )
    elif testmethod == "mannwhitney":
        print("The p-value of the Mann-Whitney U test is {}".format(results["p"]))

    else:
        raise KeyError("No recognized method for hypothesis testing was provided.")

    return results


def hypotest_grid(generator, survey, N_grid, fast, testmethod):
    params = ("f_life", "NUV_thresh")
    log = (True, True)
    bounds = np.array([[1e-3, 1.0], [10.0, 1e5]])
    bounds_null = np.array([[1e-3, 1.0]])
    features = ("max_nuv",)
    labels = ("has_O2",)

    h_nuv = Hypothesis(
        h1, bounds, params=params, features=features, labels=labels, log=log
    )
    h_nuv.h_null = Hypothesis(
        h_null,
        bounds_null,
        params=("f_O2",),
        features=features,
        labels=labels,
        log=(True,),
    )

    if fast:
        N_iter = 3
    else:
        N_iter = 16
    # f_life = np.geomspace(0.1, 1.0, N_grid)
    f_life = np.geomspace(1e-3, 1.0, N_grid)
    # f_life = np.geomspace(0.5, 1.0, N_grid)
    # f_life = (0.9,)  # test 1D hypothesis grid test
    # f_life = 0.99  # test 1D hypothesis grid test
    # NUV_thresh = np.geomspace(300.0, 380.0, N_grid)
    # NUV_thresh = np.geomspace(30.0, 3000.0, N_grid)
    # NUV_thresh = np.geomspace(200.0, 600.0, N_grid)
    NUV_thresh = np.geomspace(10.0, 1000.0, N_grid)

    from bioverse.analysis import test_hypothesis_grid

    results = test_hypothesis_grid(
        h_nuv,
        generator,
        survey,
        method=testmethod,
        # method="emcee",
        mw_alternative="two-sided",
        # method="dynesty",
        f_life=f_life,
        NUV_thresh=NUV_thresh,
        N=N_iter,
        processes=8,
        t_total=10 * 365.25,
        error_dump_filename=paths.root / "out/error_dump.txt",
    )
    return results


def get_params_past_uv(hoststars="all", **kwargs):
    """define default parameters for the past UV hypothesis test."""
    params_past_uv = {
        # "d_max": 60,        # TOO SMALL SAMPLE AND THE HYPOTHESIS TESTING GRID GETS STUCK WITHOUT AN ERROR MESSAGE
        "d_max": 75,  # TOO SMALL SAMPLE AND THE HYPOTHESIS TESTING GRID GETS STUCK WITHOUT AN ERROR MESSAGE
        "deltaT_min": 10.0,  # Myr
        # "NUV_thresh": 350.0,  # choose such that n_inhabited can't be zero
        # "NUV_thresh": 380.0,  # choose such that n_inhabited can't be zero
        "NUV_thresh": 250.0,  # choose such that n_inhabited can't be zero
        # "f_life": 0.8,
        "f_life": 1.0,
        # "f_eta": 5.0,  # Occurrence rate scaling factor (MAKE SURE SAMPLE IS LARGE ENOUGH (see above))
    }

    # replace parameters with kwargs, if any
    for key, value in kwargs.items():
        params_past_uv[key] = value

    if hoststars == "FGK":
        # exclude other spectral types
        params_past_uv["SpT"] = ["F", "G", "K"]
        params_past_uv["d_max"] = 125  # Gaia GCNS doesn't go further than 119 pc
        params_past_uv[
            "f_eta"
        ] = 6.0  # scale to obtain 100 transiting EECs in the sample

    elif hoststars == "M":
        # only M dwarf hosts
        params_past_uv["SpT"] = ["M"]
        params_past_uv["d_max"] = 42.5
        params_past_uv[
            "f_eta"
        ] = 5.0  # scale to obtain 100 transiting EECs in the sample

    elif hoststars == "all":
        # default, volume-limited
        pass

    return params_past_uv


def past_uv(
    hoststars="all",
    grid=True,
    N_grid=None,
    testmethod="mannwhitney",
    powergrid=False,
    fast=False,
    **kwargs
):
    """Test the hypothesis that life only originates on planets with a minimum past UV irradiance.

    Parameters
    ----------
    hoststars : str
        The spectral type of the host stars to consider.
        Options are 'all', 'FGK', 'M'.
    grid : bool
        Whether to perform a grid of hypothesis tests.
    N_grid : int
        The number of grid points to use.
    powergrid : bool
        Whether to run a grid of statistical power calculations.
    fast : bool
        Whether to run a fast grid for testing (less grid points)
    kwargs : dict
        Additional parameters to pass to the generator.

    Returns
    -------
    d : Table
        The sample of planets.
    grid : Table
        The grid of hypothesis tests.
    detected : Table
        The detected planets.
    data : Table
        The measurements obtained.
    nautilus : TransitSurvey object
        The transit survey.
    """

    # default parameters for planet generation
    params_past_uv = get_params_past_uv(hoststars)

    g, g_args = generate_generator(label=None, **params_past_uv)  # , **kwargs)

    if grid:
        # perform a grid of hypothesis tests
        if fast:
            N_grid = 2
        elif N_grid:
            N_grid = N_grid
        else:
            N_grid = 10
        nautilus = create_survey_nautilus()
        grid = hypotest_grid(
            g, nautilus, N_grid=N_grid, fast=fast, testmethod=testmethod
        )
        d = None
        detected = None
        data = None
    # elif powergrid:
    #     # perform a grid of statistical power calculations
    #
    #     d_max = np.linspace(10, 100, 6)
    #     nautilus = create_survey_nautilus()
    #
    #     # NUV_thresh=350.,
    #
    #     from bioverse.analysis import test_hypothesis_grid
    #
    #     # Compute the statistical power for each parameter combination
    #     power = analysis.compute_statistical_power(results, method="dlnZ", threshold=3)
    #
    #     return d, grid, detected, data, nautilus
    #
    else:
        # perform a single hypothesis test

        d = g.generate()
        # keep only 100 planets
        d = d[:100]

        dd = d.to_pandas()
        print("Total number of planets: {}".format(len(d)))
        print("Inhabited: {}".format(len(dd[dd.inhabited])))

        grid = None
        d, detected, data, nautilus = run_survey_nautilus(d)
        print("Number of planets in the sample: {}".format(len(d)))
        results = hypothesis_test(data, testmethod)
        try:
            save_var_latex("dlnZ_{}".format(hoststars), results["dlnZ"])
        except KeyError:
            save_var_latex("p_{}".format(hoststars), results["p"])

        # save some variables for the manuscript
        if hoststars == "FGK":
            save_var_latex("uv_inhabited_FGK", len(dd[dd.inhabited]))

            # general
            save_var_latex("d_max", g_args["d_max"])
            save_var_latex("M_G_max", g_args["M_G_max"])
            save_var_latex("M_st_max", g_args["M_st_max"])
            try:
                save_var_latex("f_life", params_past_uv["f_life"])
            except KeyError:
                pass
            save_var_latex("NUV_thresh", params_past_uv["NUV_thresh"])
            save_var_latex("deltaT_min", int(params_past_uv["deltaT_min"]))
            save_var_latex("uv_inhabited", len(dd[dd.inhabited]))

        elif hoststars == "M":
            save_var_latex("uv_inhabited_M", len(dd[dd.inhabited]))
        elif hoststars == "all":
            # general run (are we even still doing this)?
            pass
            # save_var_latex("d_max", g_args["d_max"])
            # save_var_latex("M_G_max", g_args["M_G_max"])
            # save_var_latex("f_life", params_past_uv["f_life"])
            # save_var_latex("NUV_thresh", params_past_uv["NUV_thresh"])
            # save_var_latex("deltaT_min", int(params_past_uv["deltaT_min"]))
            # save_var_latex("uv_inhabited", len(dd[dd.inhabited]))

    # fixed variables from semianalytical analysis
    save_var_latex("semian_Nsamp1", 10)
    save_var_latex("semian_Nsamp2", 100)
    ## Below commands potentially lead to latex error "Paragraph ended before \@dtl@stripeol was complete":
    # save_var_latex("sigma_M_st", nautilus.measurements['M_st'].precision)
    # save_var_latex("sigma_t", nautilus.measurements['age'].precision)
    return d, grid, detected, data, nautilus


@timeit
# def main(fast=False, testmethod="mannwhitney"):
def main(fast=False, testmethod="dynesty"):
    """Run the Bioverse pipeline."""
    for grid in [False, True]:
    # for grid in [True]:
    # for grid in [False]:
        # DEBUG

        # for spt in ["all", "FGK", "M"]:
        for spt in ["FGK", "M"]:
            if grid:
                print("grid run")
                if fast:
                    print("...in fast mode")

                # grid runs
                _d, grid, _detected, _data, _nautilus = past_uv(
                    hoststars=spt, grid=True, testmethod=testmethod, fast=fast
                )
                # save grid results
                with open(
                    paths.data / "pipeline/grid_flife_nuv_{}.dll".format(spt), "wb"
                ) as file:
                    dill.dump(grid, file)

            else:
                # single hypothesis test
                _d, _grid, _detected, _data, nautilus = past_uv(
                    hoststars=spt, testmethod=testmethod, grid=False
                )

                # save Bioverse objects
                with open(
                    paths.data / "pipeline/sample_{}.dll".format(spt), "wb"
                ) as file:
                    dill.dump(_d, file)
                with open(
                    paths.data / "pipeline/data_{}.dll".format(spt), "wb"
                ) as file:
                    dill.dump(_data, file)

    try:
        # save relevant Nautilus survey parameters to variables file
        save_var_latex("nautilus_max_nuv", nautilus.measurements["max_nuv"].precision)
        save_var_latex("nautilus_S", nautilus.measurements["S"].precision)
        save_var_latex("nautilus_T_eff_st", nautilus.measurements["T_eff_st"].precision)
    except:
        pass

    #
    # # run statistical power grids
    # for spt in ["FGK", "M"]:
    #     _d, powergrid, _detected, _data, _nautilus = past_uv(
    #         hoststars=spt, grid=False, testmethod=testmethod, powergrid=True
    #     )
    #
    return


if __name__ == "__main__":
    # result = timeit.timeit("main()", number=1)
    # main(fast=True)
    main(fast=False)


# # -----------------
# # SOME DEBUGGING: experiment with NUV_thresh for FGK and M stars, plot the "detections_uv" as diagnostics
#
# from matplotlib import pyplot as plt
#
# NUV_thresh_i = 250
# from detections_uv import plot_detections_uv
#
# d_FGK, grid, detected_FGK, data_FGK, nautilus = past_uv(
#     hoststars="FGK",
#     grid=False,
#     N_grid=None,
#     powergrid=False,
#     fast=False,
#     NUV_thresh=NUV_thresh_i,
# )
# fig, ax = plt.subplots(1, 1, figsize=(15, 2.0), sharex=True)
# fig, ax = plot_detections_uv(data_FGK, fig, ax, NUV_thresh_i)
# ax.set_title(f"FGK-type host stars")
# ax.text(
#     0.97,
#     0.85,
#     "FGK, {}".format(NUV_thresh_i),
#     transform=ax.transAxes,
#     horizontalalignment="right",
#     color="0.2",
# )
# plt.show()
#
# d_M, grid, detected_M, data_M, nautilus = past_uv(
#     hoststars="M",
#     grid=False,
#     N_grid=None,
#     powergrid=False,
#     fast=False,
#     NUV_thresh=NUV_thresh_i,
# )
# fig, ax = plt.subplots(1, 1, figsize=(15, 2.0), sharex=True)
# fig, ax = plot_detections_uv(data_M, fig, ax, NUV_thresh_i)
# ax.set_title(f"M-type host stars")
# ax.text(
#     0.97,
#     0.85,
#     'M, {}'.format(NUV_thresh_i),
#     transform=ax.transAxes,
#     horizontalalignment="right",
#     color="0.2",
# )
# plt.show()
