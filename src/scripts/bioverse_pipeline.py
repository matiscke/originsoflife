"""
This file contains the code to generate the datasets for the paper.
It heavily relies on Bioverse and its auxiliary functions.
"""
import pickle

import paths
from utils import *

# import pickle
import numpy as np

# Import the Generator class
from bioverse.generator import Generator
from bioverse.survey import TransitSurvey

# from bioverse.constants import CONST, DATA_DIR


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
    """define generator parameters"""
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
        g.insert_step(inject_nuv_life_correlation)
        g.insert_step(inject_biosignature)
    [g.set_arg(key, val) for key, val in g_args.items()]
    return g


def survey_nautilus(sample, t_total=10 * 365.25):
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
    """
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
    }

    # margs['t_ref'] = {'R':1.5}

    # Add the measurements to the survey
    for mkey in mkeys:
        mkwargs = {}
        for key, vals in margs.items():
            if mkey in vals:
                mkwargs[key] = vals[mkey]
        nautilus.add_measurement(mkey, **mkwargs)

    # compute yield, conduct survey
    detected = nautilus.compute_yield(sample)
    save_var_latex("N_nautilus", len(detected))
    data = nautilus.observe(detected, t_total=t_total)

    return sample, detected, data





def past_uv(**kwargs):
    """Test the hypothesis that life only originates on planets with a minimum past UV irradiance."""

    # default parameters for planet generation
    params_past_uv = {
        "d_max": 35,
        "NUV_thresh": 300.0,
        "f_life": 0.8,
        "deltaT_min": 100.0,
        'f_eta': 5.0,         # Occurrence rate scaling factor
    }



    g = generate_generator(label=None, **params_past_uv, **kwargs)
    d = g.generate()

    dd = d.to_pandas()

    print("Total number of planets: {}".format(len(d)))
    print("Inhabited: {}".format(len(dd[dd.inhabited])))

    # save some variables for the manuscript
    save_var_latex("f_life", params_past_uv["f_life"])
    save_var_latex("deltaT_min", int(params_past_uv["deltaT_min"]))
    save_var_latex("uv_inhabited", len(dd[dd.inhabited]))

    return d


# if __name__ == "__main__":
print('RUNNING BIOVERSE PIPELINE')
d = past_uv()
# save Bioverse objects
with open(paths.data / 'pipeline/sample.pkl', 'wb') as file:
    pickle.dump(d, file)