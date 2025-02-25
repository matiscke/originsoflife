# Plots and conversions for Spectral Energy Distributions (SEDs)

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import h, c, k_B
import astropy.units as u
import pandas as pd


def PlanckFunctionLambda(Temperature, lambdas):
    iArray = np.zeros(len(lambdas))
    dlamArray = np.zeros(len(lambdas))
    for i in range(0, len(lambdas) - 1):
        dlamArray[i] = lambdas[i + 1] - lambdas[i]
        iArray[i] = (2.0 * h.value * c.value ** 2 / lambdas[i] ** 5) * (
                1.0 / (np.exp(h.value * c.value / (lambdas[i] * k_B.value * Temperature)) - 1.0))
    return iArray, dlamArray


def plot_sed(ax, wavelength, flux, label, color):
    ax.plot(wavelength, flux, label=label, color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux (erg/s/cm^2/nm)')
    ax.legend()


def bb_sed(min_wl=100, max_wl=9000):
    """
    Plot the spectral energy distributions of different spectral types.

    Parameters
    ----------
    min_wl : float
        Minimum wavelength in nm.
    max_wl : float
        Maximum wavelength in nm.
    """

    wavelengths = np.geomspace(min_wl, max_wl, 100) * u.nm  # Wavelengths from 10 nm to 1000 nm
    temperatures = {'hot': 20000, 'F-type': 7000, 'G-type': 5800, 'K-type': 4500, 'M-type': 3000}

    fig, ax = plt.subplots()
    colors = {'hot': 'cyan', 'F-type': 'blue', 'G-type': 'green', 'K-type': 'orange', 'M-type': 'red'}

    for star_type, temp in temperatures.items():
        flux, _ = PlanckFunctionLambda(temp, wavelengths.to(u.m).value)
        plot_sed(ax, wavelengths.value, flux, star_type, colors[star_type])

    plt.title('Spectral Energy Distributions of Different Spectral Types')
    plt.show()


# def bb2photons():


def load_flux_data(filepath):
    df = pd.read_csv(filepath, skiprows=2, usecols=[0, 1, 2],
                     names=["Age (Myr)", "Spectral Type", "NUV Flux (erg/s/cm^2)"])
    df.dropna(inplace=True)
    df["Age (Myr)"] = pd.to_numeric(df["Age (Myr)"], errors='coerce')
    df["NUV Flux (erg/s/cm^2)"] = pd.to_numeric(df["NUV Flux (erg/s/cm^2)"], errors='coerce')
    return df


def planck_law(wavelength, temperature):
    return (2 * h * c ** 2 / wavelength ** 5) / (np.exp(h * c / (wavelength * k_B * temperature)) - 1)


def compute_blackbody_seds(temperatures, wavelengths):
    # return {spectral_type: planck_law(wavelengths, temp) for spectral_type, temp in temperatures.items()}
    return {spectral_type: PlanckFunctionLambda(temp, wavelengths)[0] for spectral_type, temp in temperatures.items()}


def normalize_seds(flux_data, blackbody_seds, wavelengths):
    flux_reference = {
        "G-type": flux_data[flux_data["Spectral Type"] == "G"]["NUV Flux (erg/s/cm^2)"].median(),
        "K-type": flux_data[flux_data["Spectral Type"] == "K"]["NUV Flux (erg/s/cm^2)"].median(),
        "early M": flux_data[flux_data["Spectral Type"] == "earlyM"]["NUV Flux (erg/s/cm^2)"].median(),
        "late M": flux_data[flux_data["Spectral Type"] == "lateM"]["NUV Flux (erg/s/cm^2)"].median(),
    }

    photon_fluxes = {}
    wl_range = np.linspace(200e-9, 280e-9, 100)
    for spectral_type, flux in blackbody_seds.items():
        energy_flux = np.interp(wl_range, wavelengths, flux)
        photon_flux = energy_flux / (h * c / wl_range)
        total_photon_flux = np.trapz(photon_flux, wl_range)
        photon_fluxes[spectral_type] = total_photon_flux

    scaling_factors = {st: flux_reference[st] / photon_fluxes[st] for st in flux_reference if st in photon_fluxes}

    normalized_seds = {st: blackbody_seds[st] * scaling_factors.get(st, 1) for st in blackbody_seds}
    return normalized_seds, scaling_factors


def plot_seds(seds, wavelengths, title="Normalized Blackbody SEDs"):
    plt.figure(figsize=(8, 6))
    for st, flux in seds.items():
        plt.plot(wavelengths * 1e9, flux / 1e-13, label=st)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Spectral Energy Flux (erg/s/cm²)")
    plt.title(title)
    plt.legend()
    plt.yscale("log")
    # plt.xlim(100, 3000)
    plt.grid(True)
    plt.show()


def main():
    flux_data = load_flux_data("../data/past-UV.csv")
    temperatures = {"G-type": 5700, "K-type": 4500, "early M": 3500, "late M": 2500}
    wavelengths = np.linspace(100e-9, 300e-9, 1000)

    blackbody_seds = compute_blackbody_seds(temperatures, wavelengths)
    normalized_seds, scaling_factors = normalize_seds(flux_data, blackbody_seds, wavelengths)
    plot_seds(normalized_seds, wavelengths)

    # df_scaling_factors = pd.DataFrame.from_dict(scaling_factors, orient="index", columns=["Scaling Factor"])
    # df_scaling_factors.to_csv("../data/scaling_factors.csv")


if __name__ == "__main__":
    # bb_sed()
    main()