# Plots and conversions for Spectral Energy Distributions (SEDs)

# import matplotlib
# matplotlib.use('Qt5Agg')  # or 'Qt5Agg', 'Qt4Agg', etc. depending on your setup
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import h, c, k_B
import astropy.units as u
import pandas as pd
import astropy.constants as const
import paths
import cmocean
from utils import save_var_latex


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
    temperatures = {'hot': 20000, 'F-type': 7000,'K-type': 4500, 'M-type': 3000}

    fig, ax = plt.subplots()
    colors = {'hot': 'cyan', 'F-type': 'blue','K-type': 'orange', 'M-type': 'red'}

    for star_type, temp in temperatures.items():
        flux, _ = PlanckFunctionLambda(temp, wavelengths.to(u.m).value)
        plot_sed(ax, wavelengths.value, flux, star_type, colors[star_type])

    plt.title('Spectral Energy Distributions of Different Spectral Types')
    plt.show()


# def bb2photons():


def load_flux_data(filepath):
    df = pd.read_csv(filepath, skiprows=2, usecols=[0, 1, 2], header=0)
                    #  names=["age", "SpT", "NUV_flux"])
    df.dropna(inplace=True)
    df["age"] = pd.to_numeric(df["age"], errors='coerce')
    df["NUV_flux"] = pd.to_numeric(df["NUV_flux"], errors='coerce')
    return df


def planck_law(wavelength, temperature):
    return (2 * h * c ** 2 / wavelength ** 5) / (np.exp(h * c / (wavelength * k_B * temperature)) - 1)


def compute_blackbody_seds(temperatures, wavelengths):
    """Compute blackbody SEDs for given temperatures and wavelengths."""
    blackbody_seds = {}
    wavelengths = wavelengths * u.m
    
    for st, temp in temperatures.items():
        temp = temp * u.K
        # Planck function in spectral energy flux density
        bb_intensity = ((2 * const.h * const.c**2 / wavelengths**5) * 
                       1 / (np.exp(const.h * const.c / (wavelengths * const.k_B * temp)) - 1)
                      ).to(u.erg/u.s/u.cm**2/u.m)
        blackbody_seds[st] = bb_intensity
    
    return blackbody_seds


def normalize_seds(flux_data, blackbody_seds, wavelengths, age, temperatures):
    # Filter flux data for specific age
    age_data = flux_data[flux_data["age"] == age]
    
    flux_reference = {
        "K-type": age_data[age_data["SpT"] == "K"]["NUV_flux"].median(),
        "early M": age_data[age_data["SpT"] == "earlyM"]["NUV_flux"].median(),
        "late M": age_data[age_data["SpT"] == "lateM"]["NUV_flux"].median(),
    }
    
    # Convert reference fluxes to astropy quantities
    flux_reference = {k: v * u.erg/u.s/u.cm**2 for k, v in flux_reference.items()}
    
    normalized_seds = {}
    scaling_factors = {}
    
    wavelengths = wavelengths * u.m
    
    for spectral_type, temp in temperatures.items():
        temp = temp * u.K
        blackbody_sed = blackbody_seds[spectral_type] * u.erg/u.s/u.cm**2/u.m
        
        # Calculate scaling factor using UV range
        uv_mask = (wavelengths >= 200 * u.nm) & (wavelengths <= 280 * u.nm)
        uv_flux_bb = np.trapz(blackbody_sed[uv_mask], wavelengths[uv_mask])
        scaling_factor = flux_reference[spectral_type] / uv_flux_bb

        normalized_seds[spectral_type] = blackbody_sed * scaling_factor
        scaling_factors[spectral_type] = f"{scaling_factor:.2e}"
    
    # Calculate photon number flux density for the UV range
    photon_fluxes = {}
    uv_mask = (wavelengths >= 200 * u.nm) & (wavelengths <= 280 * u.nm)
    
    for st, flux in normalized_seds.items():
        # Energy per photon: E = hc/λ
        photon_energy = (const.h * const.c / wavelengths[uv_mask]).to(u.erg)
        # Convert energy flux to photon flux density
        # First convert flux to per-nm
        flux_per_nm = flux[uv_mask].to(u.erg/u.s/u.cm**2/u.nm)
        # Then divide by photon energy to get photons
        photon_flux_density = (flux_per_nm / photon_energy) * u.photon
        # Take the mean value over the UV range
        mean_flux = np.mean(photon_flux_density)
        if np.isnan(mean_flux):
            photon_fluxes[st] = 0.0 * u.photon/u.s/u.cm**2/u.nm
        else:
            photon_fluxes[st] = mean_flux
    
    return normalized_seds, scaling_factors, photon_fluxes


def plot_seds(seds_by_age, wavelengths):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    wavelengths = wavelengths * u.m
    
    # First pass to determine global y-axis limits
    y_min, y_max = float('inf'), float('-inf')
    for seds in seds_by_age.values():
        for flux in seds.values():
            y_min = min(y_min, np.min(flux.to(u.erg/u.s/u.cm**2/u.m).value / 1e-13))
            y_max = max(y_max, np.max(flux.to(u.erg/u.s/u.cm**2/u.m).value / 1e-13))
    
    for idx, (age, seds) in enumerate(seds_by_age.items()):
        ax = axes[idx]
        for st, flux in seds.items():
            ax.plot(wavelengths.to(u.nm).value, 
                   flux.to(u.erg/u.s/u.cm**2/u.m).value / 1e-13, 
                   label=st, linewidth=2)
        ax.axvspan(200, 280, alpha=0.2, color='green', label='200-280 nm')
        ax.set_xlabel("Wavelength (nm)", fontsize=16)
        ax.set_ylabel("Normalized Energy Flux (erg/s/cm²)", fontsize=16)
        ax.set_title(f"Normalized Blackbody SEDs - Age: {age} Myr", fontsize=18, pad=15)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove the empty subplot
    axes[-1].remove()
    plt.tight_layout()
    plt.show()


def plot_photon_flux(photon_fluxes_by_age):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(list(photon_fluxes_by_age.values())[0].keys()))
    width = 0.15
    
    # Create colormap for ages
    ages = list(photon_fluxes_by_age.keys())
    colors = cmocean.cm.haline(np.linspace(0, .93, len(ages)))
    
    for i, ((age, photon_fluxes), color) in enumerate(zip(photon_fluxes_by_age.items(), colors)):
        values = [v.value for v in photon_fluxes.values()]
        ax.bar(x + i*width, values, width, label=f'{age} Myr', color=color)
    
    # Add Rimmer threshold line and uncertainty
    threshold = 6.8e10
    uncertainty = 3.6e10
    ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.8)
    ax.fill_between([-width, x[-1] + width*5], 
                    threshold - uncertainty, 
                    threshold + uncertainty, 
                    color='darkgray', alpha=0.4)
    ax.annotate('Rimmer et al. 2018\n\nthreshold surface flux',
                xy=(x[-1] + width*5, threshold),
                xytext=(10, 0), textcoords='offset points',
                va='center')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel("Spectral Type", fontsize=16)
    ax.set_ylabel("Photon Number Flux Density (photons/s/cm²/nm)", fontsize=16)
    ax.set_title("Mean UV Photon Flux Density (200-280 nm)", fontsize=18, pad=15)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(list(list(photon_fluxes_by_age.values())[0].keys()), fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yscale("log")
    
    # Move legend outside
    ax.legend(fontsize=14, bbox_to_anchor=(.95, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.savefig(paths.figures / 'photon_flux.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_number_flux_seds(seds_by_age, wavelengths):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    wavelengths = wavelengths * u.m
    
    # Set up spectral density equivalencies
    equiv = u.spectral_density(wavelengths)
    
    # First pass to determine global y-axis limits
    y_min, y_max = float('inf'), float('-inf')
    for seds in seds_by_age.values():
        for flux in seds.values():
            # Convert energy flux to photon flux using spectral equivalencies
            photon_flux = flux.to(u.photon/u.s/u.cm**2/u.m, equivalencies=equiv)
            # Convert from per meter to per nm
            photon_flux = photon_flux.to(u.photon/u.s/u.cm**2/u.nm)
            y_min = min(y_min, np.min(photon_flux.value))
            y_max = max(y_max, np.max(photon_flux.value))
    
    for idx, (age, seds) in enumerate(seds_by_age.items()):
        ax = axes[idx]
        for st, flux in seds.items():
            # Convert energy flux to photon flux using spectral equivalencies
            photon_flux = flux.to(u.photon/u.s/u.cm**2/u.m, equivalencies=equiv)
            # Convert from per meter to per nm
            photon_flux = photon_flux.to(u.photon/u.s/u.cm**2/u.nm)
            ax.plot(wavelengths.to(u.nm).value, 
                   photon_flux.value,
                   label=st, linewidth=2)
        ax.axvspan(200, 280, alpha=0.2, color='green', label='200-280 nm')
        ax.set_xlabel("Wavelength (nm)", fontsize=16)
        ax.set_ylabel("Photon Number Flux Density\n(photons/s/cm²/nm)", fontsize=16)
        ax.set_title(f"Normalized Photon Flux Density - Age: {age} Myr", fontsize=18, pad=15)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove the empty subplot
    axes[-1].remove()
    plt.tight_layout()
    plt.savefig(paths.figures / 'normalized_photon_seds.pdf', bbox_inches='tight')
    plt.show()


def convert_rimmer_threshold():
    """Convert the Rimmer et al. (2018) photon flux threshold to energy flux.
    
    The threshold was determined using mercury lamps which emit >90% of their energy at 254 nm.
    Therefore, we assume all photons have wavelength 254 nm for the conversion.
    
    Rimmer et al. (2018) threshold:
    - Photon flux: 6.8±3.6 × 10¹⁰ photons cm⁻² s⁻¹ nm⁻¹ integrated from 200-280nm
    """
    # Define threshold and uncertainty (single source for these values)
    threshold_photon = 6.8e10 * u.photon/u.cm**2/u.s/u.nm  # photons cm⁻² s⁻¹ nm⁻¹
    uncertainty_photon = 3.6e10 * u.photon/u.cm**2/u.s/u.nm
    
    # Integration range
    delta_lambda = (280 - 200) * u.nm
    
    # Total integrated photon flux
    total_photon_flux = threshold_photon * delta_lambda
    total_photon_uncertainty = uncertainty_photon * delta_lambda
    
    # Set up spectral equivalency at 254 nm
    wavelength = 254 * u.nm
    equiv = u.spectral_density(wavelength)
    
    # Convert photon flux to energy flux using spectral equivalency
    energy_flux = total_photon_flux.to(u.erg/u.cm**2/u.s, equivalencies=equiv)
    energy_uncertainty = total_photon_uncertainty.to(u.erg/u.cm**2/u.s, equivalencies=equiv)
    
    # Print results
    print(f"Rimmer et al. (2018) threshold converted to energy flux:")
    print(f"Threshold: {energy_flux:.2e}")
    print(f"Uncertainty: {energy_uncertainty:.2e}")
    
    # Save to variables.dat with appropriate rounding
    # The input has 2 significant figures, so we'll round to 2 significant figures
    save_var_latex('rimmer_threshold', f"{energy_flux.value:.0f}")
    save_var_latex('rimmer_uncertainty', f"{energy_uncertainty.value:.0f}")
    
    return energy_flux, energy_uncertainty


def plot_energy_flux(flux_data):
    """Plot energy flux version of the photon flux bar plot using original integrated energy flux data."""
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(flux_data["SpT"].unique()))
    width = 0.15
    
    # Create colormap for ages (using a different colormap)
    ages = flux_data["age"].unique()
    colors = cmocean.cm.thermal(np.linspace(0, .93, len(ages)))
    
    for i, age in enumerate(ages):
        if pd.isna(age):
            continue
        # Filter the flux data for the current age
        age_data = flux_data[flux_data["age"] == age]
        
        # Get integrated energy flux values
        integrated_fluxes = age_data['NUV_flux'].values * u.erg/u.s/u.cm**2
        
        # Convert to energy flux (no need for nm conversion since we are using integrated flux)
        values = [v.value for v in integrated_fluxes]
        ax.bar(x + i*width, values, width, label=f'{age} Myr', color=colors[i])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel("Spectral Type", fontsize=16)
    ax.set_ylabel("Integrated Energy Flux (erg/s/cm²)", fontsize=16)
    ax.set_title("Mean UV Integrated Energy Flux (200-280 nm)", fontsize=18, pad=15)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(flux_data["SpT"].unique(), fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yscale("log")
    
    # Move legend outside
    ax.legend(fontsize=14, bbox_to_anchor=(.95, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.savefig(paths.figures / 'integrated_energy_flux.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def compare_dynamic_range(scaling_factors_by_age, photon_fluxes_by_age, flux_data):
    # Extract maximum values for K-type and late M
    max_photon_flux_k = max(photon_fluxes_by_age[age]["K-type"] for age in photon_fluxes_by_age)
    max_photon_flux_late_m = max(photon_fluxes_by_age[age]["late M"] for age in photon_fluxes_by_age)

    max_integrated_flux_k = flux_data[flux_data["SpT"] == "K"]["NUV_flux"].max()
    max_integrated_flux_late_m = flux_data[flux_data["SpT"] == "lateM"]["NUV_flux"].max()

    # Calculate ratios
    photon_flux_ratio = max_photon_flux_late_m / max_photon_flux_k
    integrated_flux_ratio = max_integrated_flux_late_m / max_integrated_flux_k

    print(f"Photon Flux Density Ratio (late M / K-type): {photon_flux_ratio:.2f}")
    print(f"Integrated Energy Flux Ratio (late M / K-type): {integrated_flux_ratio:.2f}")

    # Save ratios to variables.dat
    save_var_latex('photon_flux_ratio', f"{photon_flux_ratio:.2f}")
    save_var_latex('integrated_flux_ratio', f"{integrated_flux_ratio:.2f}")

    return photon_flux_ratio, integrated_flux_ratio


def plot_combined_fluxes(photon_fluxes_by_age, flux_data, photon_flux_ratio, integrated_flux_ratio):
    """Plot photon flux and energy flux side by side."""
    plt.rcParams.update({'font.size': 16})  # Increase the base font size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x = np.arange(len(list(photon_fluxes_by_age.values())[0].keys()))
    width = 0.15
    
    # Create colormap for ages
    ages = list(photon_fluxes_by_age.keys())
    colors = cmocean.cm.haline(np.linspace(0, .93, len(ages)))
    
    # Left plot: Photon flux
    for i, ((age, photon_fluxes), color) in enumerate(zip(photon_fluxes_by_age.items(), colors)):
        values = [v.value for v in photon_fluxes.values()]
        ax1.bar(x + i*width, values, width, label=f'{age} Myr', color=color)
    
    # Add Rimmer threshold line and uncertainty
    threshold = 6.8e10
    uncertainty = 3.6e10
    ax1.axhline(y=threshold, color='black', linestyle='--', alpha=0.8)
    ax1.fill_between([-width, x[-1] + width*5], 
                    threshold - uncertainty, 
                    threshold + uncertainty, 
                    color='darkgray', alpha=0.4)
    ax1.annotate('Rimmer et al. 2018\n\nthreshold surface flux',
                xy=(x[-1] + width*5, threshold),
                xytext=(10, 0), textcoords='offset points',
                va='center')
    
    # Right plot: Energy flux
    for i, age in enumerate(ages):
        age_data = flux_data[flux_data["age"] == age]
        integrated_fluxes = age_data['NUV_flux'].values * u.erg/u.s/u.cm**2
        values = [v.value for v in integrated_fluxes]
        ax2.bar(x + i*width, values, width, color=colors[i])
    
    # Common formatting
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(x + width*2)
        ax.set_xticklabels(list(list(photon_fluxes_by_age.values())[0].keys()), fontsize=16)  # Increase x-tick font size
        ax.tick_params(axis='y', labelsize=16)  # Increase y-tick font size
        ax.set_yscale("log")
        ax.set_xlabel("Spectral Type", fontsize=16)  # Increase x-label font size
    
    # Specific labels
    ax1.set_ylabel("Photon Number Flux Density\n(photons/s/cm²/nm)", fontsize=16)  # Increase y-label font size
    ax2.set_ylabel("Integrated Energy Flux\n(erg/s/cm²)", fontsize=16)  # Increase y-label font size
    
    # Title for entire figure
    fig.suptitle("UV Flux Evolution (200-280 nm)", fontsize=18, y=1.05)  # Increase title font size
    
    # Single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.99, 0.9), loc='center left', fontsize=16)  # Increase legend font size
    
    # Draw brackets and annotate ratios
    # Photon flux plot
    max_photon_flux_k = max(photon_fluxes_by_age[age]["K-type"].value for age in photon_fluxes_by_age)
    max_photon_flux_late_m = max(photon_fluxes_by_age[age]["late M"].value for age in photon_fluxes_by_age)

    # Energy flux plot
    max_integrated_flux_k = flux_data[flux_data["SpT"] == "K"]["NUV_flux"].max()
    max_integrated_flux_late_m = flux_data[flux_data["SpT"] == "lateM"]["NUV_flux"].max()

    # Add shaded regions behind bars
    k_idx = list(list(photon_fluxes_by_age.values())[0].keys()).index("K-type")
    late_m_idx = list(list(photon_fluxes_by_age.values())[0].keys()).index("late M")
    
    # Shade photon flux plot and add text with flux ratios
    ax1.fill_between([-width/2, x[k_idx] + width/2], 
                    [max_photon_flux_k, max_photon_flux_k], [max_photon_flux_k, max_photon_flux_k],
                    color='lightgray', alpha=0.75, zorder=1)
    ax1.text(x[k_idx], max_photon_flux_k + (max_photon_flux_late_m - max_photon_flux_k)/2, f'ratio: {photon_flux_ratio:.2f}', ha='left', va='top', zorder=2, alpha=0.8)
    
    ax1.fill_between([-width/2, x[late_m_idx] + width/2],
                    [max_photon_flux_late_m, max_photon_flux_late_m], [max_photon_flux_late_m, max_photon_flux_late_m], 
                    color='lightgray', alpha=0.75, zorder=1)

    # Shade energy flux plot and add text with flux ratios
    ax2.fill_between([-width/2, x[k_idx] + width/2],
                    [max_integrated_flux_k, max_integrated_flux_k], [max_integrated_flux_k, max_integrated_flux_k],
                    color='lightgray', alpha=0.75, zorder=1)
    ax2.text(x[k_idx], max_integrated_flux_k + (max_integrated_flux_late_m - max_integrated_flux_k)/2, f'ratio: {integrated_flux_ratio:.2f}', ha='left', va='top', zorder=2, alpha=0.8)
    
    ax2.fill_between([-width/2, x[late_m_idx] + width/2],
                    [max_integrated_flux_late_m, max_integrated_flux_late_m], [max_integrated_flux_late_m, max_integrated_flux_late_m],
                    color='lightgray', alpha=0.75, zorder=1)

    plt.tight_layout()
    plt.savefig(paths.figures / 'combined_fluxes.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    flux_data = load_flux_data("../data/past-UV.csv")
    temperatures = { "K-type": 4500, "early M": 3500, "late M": 2500}
    wavelengths = np.linspace(190e-9, 300e-9, 1000)
    ages = flux_data.age.unique()

    blackbody_seds = compute_blackbody_seds(temperatures, wavelengths)
    
    normalized_seds_by_age = {}
    scaling_factors_by_age = {}
    photon_fluxes_by_age = {}
    
    for age in ages:
        normalized_seds, scaling_factors, photon_fluxes = normalize_seds(flux_data, blackbody_seds, wavelengths, age, temperatures)
        normalized_seds_by_age[age] = normalized_seds
        scaling_factors_by_age[age] = scaling_factors
        photon_fluxes_by_age[age] = photon_fluxes
    
    plot_seds(normalized_seds_by_age, wavelengths)
    plot_number_flux_seds(normalized_seds_by_age, wavelengths)
    plot_photon_flux(photon_fluxes_by_age)
    plot_energy_flux(flux_data)

    # Compare dynamic range and get ratios
    photon_flux_ratio, integrated_flux_ratio = compare_dynamic_range(scaling_factors_by_age, photon_fluxes_by_age, flux_data)

    # Plot the combined fluxes with ratios
    plot_combined_fluxes(photon_fluxes_by_age, flux_data, photon_flux_ratio, integrated_flux_ratio)

    # Convert and save Rimmer threshold
    rimmer_energy_flux, rimmer_energy_uncertainty = convert_rimmer_threshold()

    # Save results for each age
    for age in ages:
        df_scaling_factors = pd.DataFrame.from_dict(scaling_factors_by_age[age], orient="index", columns=["Scaling Factor"])
        df_scaling_factors.to_csv(f"../data/scaling_factors_{age}Myr.csv")

        df_photon_flux = pd.DataFrame.from_dict(
            {k: v.value for k, v in photon_fluxes_by_age[age].items()}, 
            orient="index", 
            columns=["Photon Number Flux Density (photons/s/cm²/nm)"]
        )
        df_photon_flux.to_csv(f"../data/photon_flux_{age}Myr.csv")


if __name__ == "__main__":
    # bb_sed()
    main()