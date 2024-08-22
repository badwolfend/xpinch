import numpy as np
import matplotlib.pyplot as plt

def calculate_density_fluctuation(temperature, density, length_scale, epsilon):
    """Calculate the density fluctuation using turbulent scaling."""
    # Constants
    k_B = 1.38e-23  # Boltzmann constant, J/K
    m_i = 4.48e-26  # Mass of an aluminum ion, kg
    gamma = 5/3  # Adiabatic index for monatomic gas
    
    # Sound speed calculation
    c_s = np.sqrt(gamma * k_B * temperature / m_i)
    
    # Turbulent velocity fluctuation at the given length scale
    delta_u_l = epsilon**(1/3) * length_scale**(1/3)
    
    # Density fluctuation scaling with Mach number (delta u / c_s)
    delta_rho_l = density * (delta_u_l / c_s)
    
    return delta_rho_l
# Constants
wavelength = 532e-9  # Laser wavelength in meters (532 nm)
length_scales = np.logspace(-8, -3, 100)  # Range of length scales from 10 nm to 1 mm
epsilon = 1e18  # Energy dissipation rate in W/kg
density = 1e21  # Electron density in m^-3
mass_density = density * 4.48e-26  # Mass density in kg/m^3 for aluminum plasma
k_B = 1.38e-23  # Boltzmann constant, J/K

# Critical density for the given wavelength
m_e = 9.11e-31  # Electron mass in kg
e = 1.6e-19  # Elementary charge in C
epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
omega = 2 * np.pi * 3e8 / wavelength  # Laser angular frequency
n_critical = epsilon_0 * m_e * omega**2 / e**2  # Critical electron density
rho_critical = n_critical * 4.48e-26  # Critical mass density in kg/m^3

# Define the temperature range in electron volts (eV)
temperatures_eV = np.linspace(10, 1000, 10)  # Range from 10 eV to 1000 eV
temperatures_K = temperatures_eV * (e/k_B)  # Convert eV to Kelvin

# Recalculate the refractive index fluctuations for each temperature
plt.figure(figsize=(10, 6))

for T_eV, T_K in zip(temperatures_eV, temperatures_K):
    refractive_index_fluctuations = []
    scale_ratios = []

    for l in length_scales:
        delta_rho_l = calculate_density_fluctuation(T_K, mass_density, l, epsilon)
        delta_n = delta_rho_l / (2 * rho_critical)
        refractive_index_fluctuations.append(delta_n)
        scale_ratios.append(l / wavelength)

    refractive_index_fluctuations = np.array(refractive_index_fluctuations)
    scale_ratios = np.array(scale_ratios)

    plt.loglog(scale_ratios, refractive_index_fluctuations, label=f"T = {T_eV:.0f} eV")

# Plot configuration
plt.axvline(x=1, color='r', linestyle='--', label="l = λ")
plt.xlabel('Length Scale / Wavelength')
plt.ylabel('Refractive Index Fluctuation (δn)')
plt.title('Refractive Index Fluctuation vs. Length Scale Ratio for Different Temperatures')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()