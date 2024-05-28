"""
This file contains the code to generate the datasets for the paper.
It heavily relies on Bioverse and its auxiliary functions.
"""
# import pickle
import dill
import timeit

import paths
from utils import *

# import pickle
import numpy as np

# Import the Generator class
from bioverse.generator import Generator
from bioverse.survey import TransitSurvey

from bioverse.hypothesis import Hypothesis

# from bioverse.constants import CONST, DATA_DIR


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
    data = nautilus.observe(detected)  # commented out for now because of Bioverse's issue #45: , t_total=t_total)

    # print(data['max_nuv'][:10])

    return sample, detected, data, nautilus


# hypothesis tests
def h1(theta, X):
    f_life, NUV_thresh = theta
    return f_life * (X >= NUV_thresh)


def h_null(theta, X):
    shape = (np.shape(X)[0], 1)
    return np.full(shape, theta)


def hypothesis_test(data, method="dynesty"):
    """Perform a single hypothesis test on the data."""
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

    results = h_nuv.fit(data, method=method)

    print(
        "The evidence in favor of the hypothesis is: dlnZ = {:.1f} (corresponds to p = {:.1E})".format(
            results["dlnZ"], np.exp(-results["dlnZ"])
        )
    )
    return results


def hypotest_grid(generator, survey, N_grid, fast):
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
        N_iter = 2
    else:
        N_iter = 6
    # f_life = np.geomspace(0.1, 1.0, N_grid)
    f_life = np.geomspace(0.2, 1.0, N_grid)
    # f_life = (0.9,)  # test 1D hypothesis grid test
    # f_life = 0.99  # test 1D hypothesis grid test
    # NUV_thresh = np.logspace(1, 2.5, N_grid)
    NUV_thresh = np.geomspace(300.0, 380.0, N_grid)

    # NUV_thresh=350.,

    from bioverse.analysis import test_hypothesis_grid

    results = test_hypothesis_grid(
        h_nuv,
        generator,
        survey,
        # method="mannwhitney",
        method="dynesty",
        f_life=f_life,
        NUV_thresh=NUV_thresh,
        N=N_iter,
        processes=8,
        t_total=10 * 365.25,
    )
    return results


def past_uv(grid=True, N_grid=None, fast=False, **kwargs):
    """Test the hypothesis that life only originates on planets with a minimum past UV irradiance."""

    # default parameters for planet generation
    params_past_uv = {
        # "d_max": 60,        # TOO SMALL SAMPLE AND THE HYPOTHESIS TESTING GRID GETS STUCK WITHOUT AN ERROR MESSAGE
        "d_max": 75,  # TOO SMALL SAMPLE AND THE HYPOTHESIS TESTING GRID GETS STUCK WITHOUT AN ERROR MESSAGE
        "deltaT_min": 10.0,  # Myr
        "NUV_thresh": 350.0,  # choose such that n_inhabited can't be zero
        # "NUV_thresh": 200.0,
        "f_life": 0.8,
        "f_eta": 5.0,  # Occurrence rate scaling factor (MAKE SURE SAMPLE IS LARGE ENOUGH (see above))
    }

    # replace parameters with kwargs, if any
    for key, value in kwargs.items():
        params_past_uv[key] = value

    g, g_args = generate_generator(label=None, **params_past_uv, **kwargs)
    d = g.generate()
    dd = d.to_pandas()

    print("Total number of planets: {}".format(len(d)))
    print("Inhabited: {}".format(len(dd[dd.inhabited])))

    # d.evolve()

    if grid:
        # perform a grid of hypothesis tests
        if fast:
            N_grid = 2
        elif N_grid:
            N_grid = N_grid
        else:
            N_grid = 6
        nautilus = create_survey_nautilus()
        grid = hypotest_grid(g, nautilus, N_grid=N_grid, fast=fast)
        detected = None
        data = None
    else:
        # perform a single hypothesis test
        grid = None
        d, detected, data, nautilus = run_survey_nautilus(d)
        _ = hypothesis_test(data)

    print("Number of planets in the sample: {}".format(len(d)))

    # save some variables for the manuscript
    save_var_latex("d_max", g_args["d_max"])
    save_var_latex("M_G_max", g_args["M_G_max"])

    save_var_latex("f_life", params_past_uv["f_life"])
    save_var_latex("NUV_thresh", params_past_uv["NUV_thresh"])
    save_var_latex("deltaT_min", int(params_past_uv["deltaT_min"]))
    save_var_latex("uv_inhabited", len(dd[dd.inhabited]))

    # fixed variables from semianalytical analysis
    save_var_latex("semian_Nsamp1", 10)
    save_var_latex("semian_Nsamp2", 100)
    ## Below commands potentially lead to latex error "Paragraph ended before \@dtl@stripeol was complete":
    # save_var_latex("sigma_M_st", nautilus.measurements['M_st'].precision)
    # save_var_latex("sigma_t", nautilus.measurements['age'].precision)
    return d, grid, detected, data, nautilus


def main(fast=True):
    """Run the Bioverse pipeline."""
    print("RUNNING BIOVERSE PIPELINE")

    d, _grid, detected, data, nautilus = past_uv(grid=False)
    _d, grid, _detected, _data, _nautilus = past_uv(fast=fast)


    # save Bioverse objects
    with open(paths.data / "pipeline/sample.dll", "wb") as file:
        dill.dump(d, file)
    with open(paths.data / "pipeline/data.dll", "wb") as file:
        dill.dump(data, file)
    with open(paths.data / "pipeline/grid_flife_nuv.dll", "wb") as file:
        dill.dump(grid, file)

    return


if __name__ == "__main__":
    # result = timeit.timeit("main()", number=1)
    result = main()

    # wait = input("PRESS ENTER TO CONTINUE.")
