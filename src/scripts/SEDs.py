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


def normalize_seds(flux_data, blackbody_seds, wavelengths, age):
    # Filter flux data for specific age
    age_data = flux_data[flux_data["Age (Myr)"] == age]
    
    flux_reference = {
        "G-type": age_data[age_data["Spectral Type"] == "G"]["NUV Flux (erg/s/cm^2)"].median(),
        "K-type": age_data[age_data["Spectral Type"] == "K"]["NUV Flux (erg/s/cm^2)"].median(),
        "early M": age_data[age_data["Spectral Type"] == "earlyM"]["NUV Flux (erg/s/cm^2)"].median(),
        "late M": age_data[age_data["Spectral Type"] == "lateM"]["NUV Flux (erg/s/cm^2)"].median(),
    }

    photon_fluxes = {}
    wl_range = np.linspace(200e-9, 280e-9, 100)
    for spectral_type, flux in blackbody_seds.items():
        energy_flux = np.interp(wl_range, wavelengths, flux)
        photon_flux = energy_flux / (h.value * c.value / wl_range)
        total_photon_flux = np.trapz(photon_flux, wl_range)
        photon_fluxes[spectral_type] = float(total_photon_flux)

    scaling_factors = {st: flux_reference[st] / photon_fluxes[st] for st in flux_reference if st in photon_fluxes}

    normalized_seds = {st: blackbody_seds[st] * scaling_factors.get(st, 1) for st in blackbody_seds}

    # Calculate photon number flux density (per nm) for the UV range
    photon_fluxes = {}
    uv_mask = (wavelengths >= 200e-9) & (wavelengths <= 280e-9)
    wavelength_nm = wavelengths[uv_mask] * 1e9  # Convert to nm
    
    for st, flux in normalized_seds.items():
        # Energy per photon: E = hc/λ
        photon_energy = (h * c) / wavelengths[uv_mask]
        # Convert energy flux to photon flux density (per nm)
        photon_flux_density = flux[uv_mask] / photon_energy / 1e9  # Divide by 1e9 to convert from per m to per nm
        # Take the mean value over the UV range
        photon_fluxes[st] = f"{np.mean(photon_flux_density):.2e}"
    
    return normalized_seds, scaling_factors, photon_fluxes


def plot_seds(seds_by_age, wavelengths):
    plt.rcParams.update({'font.size': 14})  # Set base font size
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # First pass to determine global y-axis limits
    y_min, y_max = float('inf'), float('-inf')
    for seds in seds_by_age.values():
        for flux in seds.values():
            y_min = min(y_min, np.min(flux / 1e-13))
            y_max = max(y_max, np.max(flux / 1e-13))
    
    for idx, (age, seds) in enumerate(seds_by_age.items()):
        ax = axes[idx]
        for st, flux in seds.items():
            ax.plot(wavelengths * 1e9, flux / 1e-13, label=st, linewidth=2)
        ax.axvspan(200e-9 * 1e9, 280e-9 * 1e9, alpha=0.2, color='green', label='200-280 nm')
        ax.set_xlabel("Wavelength (nm)", fontsize=16)
        ax.set_ylabel("Normalized Spectral Energy Flux (erg/s/cm²)", fontsize=16)
        ax.set_title(f"Normalized Blackbody SEDs - Age: {age} Myr", fontsize=18, pad=15)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove the empty subplot
    axes[-1].remove()
    plt.tight_layout()
    plt.show()


def plot_photon_flux(photon_fluxes_by_age):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(list(photon_fluxes_by_age.values())[0].keys()))
    width = 0.15
    
    for i, (age, photon_fluxes) in enumerate(photon_fluxes_by_age.items()):
        values = [float(v) for v in photon_fluxes.values()]
        plt.bar(x + i*width, values, width, label=f'{age} Myr')
    
    plt.xlabel("Spectral Type", fontsize=16)
    plt.ylabel("Photon Number Flux Density (photons/s/cm²/nm)", fontsize=16)
    plt.title("Mean UV Photon Flux Density (200-280 nm)", fontsize=18, pad=15)
    plt.xticks(x + width*2, list(list(photon_fluxes_by_age.values())[0].keys()), fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    plt.legend(fontsize=14)
    plt.show()


def main():
    flux_data = load_flux_data("../data/past-UV.csv")
    temperatures = {"G-type": 5700, "K-type": 4500, "early M": 3500, "late M": 2500}
    wavelengths = np.linspace(190e-9, 300e-9, 1000)
    ages = [16.5, 43.5, 150.0, 650.0, 5000.0]

    blackbody_seds = compute_blackbody_seds(temperatures, wavelengths)
    
    normalized_seds_by_age = {}
    scaling_factors_by_age = {}
    photon_fluxes_by_age = {}
    
    for age in ages:
        normalized_seds, scaling_factors, photon_fluxes = normalize_seds(flux_data, blackbody_seds, wavelengths, age)
        normalized_seds_by_age[age] = normalized_seds
        scaling_factors_by_age[age] = scaling_factors
        photon_fluxes_by_age[age] = photon_fluxes
    
    plot_seds(normalized_seds_by_age, wavelengths)
    plot_photon_flux(photon_fluxes_by_age)

    # Save results for each age
    for age in ages:
        df_scaling_factors = pd.DataFrame.from_dict(scaling_factors_by_age[age], orient="index", columns=["Scaling Factor"])
        df_scaling_factors.to_csv(f"../data/scaling_factors_{age}Myr.csv")

        df_photon_flux = pd.DataFrame.from_dict(photon_fluxes_by_age[age], orient="index", columns=["Photon Number Flux"])
        df_photon_flux.to_csv(f"../data/photon_flux_{age}Myr.csv")


if __name__ == "__main__":
    # bb_sed()
    main()