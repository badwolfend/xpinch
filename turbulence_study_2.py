import numpy as np

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

import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 532e-9  # Laser wavelength in meters
L = 100*wavelength  # Distance traveled by the laser in meters
length_scale = wavelength  # Fluctuation length scale in meters
epsilon = 1e10  # Energy dissipation rate in W/kg

# Critical density for the given wavelength
m_e = 9.11e-31  # Electron mass in kg
e = 1.6e-19  # Elementary charge in C
k_B = 1.38e-23  # Boltzmann constant, J/K
epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
omega = 2 * np.pi * 3e8 / wavelength  # Laser angular frequency
n_critical = epsilon_0 * m_e * omega**2 / e**2  # Critical electron density
rho_critical = n_critical * 4.48e-26  # Critical mass density in kg/m^3
densities = np.linspace(n_critical/100, n_critical/10, 1000)  # Range of electron densities in m^-3
temperatures = np.linspace(1e4, 1e6, 1000)  # Range of temperatures in Kelvin
delta_n_array = np.zeros_like(densities)
# Initialize arrays to store results
phase_shifts = np.zeros((len(temperatures), len(densities)))
beam_deviations = np.zeros((len(temperatures), len(densities)))

# Calculate phase shift and beam deviation for each combination of density and temperature
for i, T in enumerate(temperatures):
    for j, n_e in enumerate(densities):
        mass_density = n_e * 4.48e-26  # Mass density in kg/m^3 for aluminum plasma
        delta_rho_l = calculate_density_fluctuation(T, mass_density, length_scale, epsilon)
        delta_n = delta_rho_l / (2 * rho_critical)
        delta_n_array[j] = delta_n
        phase_shift = (2 * np.pi * delta_n * L) / wavelength
        beam_deviation = (delta_n * L) / length_scale
        phase_shifts[i, j] = phase_shift
        beam_deviations[i, j] = beam_deviation * (180 / np.pi)  # Convert to degrees

# Plot phase shift
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# plt.contourf(densities / n_critical, (k_B/e)*temperatures, phase_shifts, levels=1000, cmap='RdBu')
plt.contourf(delta_n_array, (k_B/e)*temperatures, phase_shifts, levels=1000, cmap='RdBu')

plt.colorbar(label='Phase Shift (radians)')
plt.xlabel('Electron Density / Critical Density')
plt.ylabel('Temperature (eV)')
plt.title('Phase Shift')

# Plot beam deviation
plt.subplot(1, 2, 2)
plt.contourf(densities / n_critical, (k_B/e)*temperatures, beam_deviations, levels=1000, cmap='RdBu')
plt.colorbar(label='Beam Deviation (degrees)')
plt.xlabel('Electron Density / Critical Density')
plt.ylabel('Temperature (eV)')
plt.title('Beam Deviation')

plt.tight_layout()
plt.show()

