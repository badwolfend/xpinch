import numpy as np
import matplotlib.pyplot as plt

def refractive_index_fluctuation(dne_ne0, n_critical, ne0):
    """Calculate the refractive index fluctuation from the number density fluctuation."""
    return dne_ne0 *ne0/ (2 * n_critical)

def phase_shift(delta_n, L, wavelength):
    """Calculate the phase shift due to the refractive index fluctuation."""
    return (2 * np.pi * delta_n * L) / wavelength

def beam_deviation(delta_n, L, wavelength):
    """Calculate the beam deviation due to the refractive index gradient."""
    return delta_n * L / wavelength

def sound_speed(temperature, gamma=5/3, NN=1):
    """Calculate the sound speed in the plasma."""
    return np.sqrt(gamma * k_B * temperature / (NN*m_p))

def density_fluctuation(M, temperature, L, wavelength, NN=1):
    """Estimate the density fluctuation for a given Mach number and temperature."""
    c_s = sound_speed(temperature, NN=NN)
    delta_u = M * c_s
    epsilon = (delta_u**3) / L  # Energy dissipation rate (approximation)
    delta_u_lambda = epsilon**(1/3) * wavelength**(1/3)  # Velocity fluctuation at small scale
    drho_rho0 = delta_u_lambda / c_s  # Relative density fluctuation
    return drho_rho0

def plot_density_fluctuation(L_range, wavelength, mach_range, T, rho_0, critical_density, NN=1):
    """Plot the density fluctuation drho/rho_0 on a M vs. T 2D color plot."""
    M_values = np.linspace(*mach_range, 100)
    L_values = np.linspace(*L_range, 100)

    drho_rho0 = np.zeros((len(L_values), len(M_values)))
    phase_shifts = np.zeros((len(L_values), len(M_values)))
    beam_deviations = np.zeros((len(L_values), len(M_values)))

    for i, L in enumerate(L_values):
        # print(f"Calculating density fluctuation for T = {T} K...")
        for j, M in enumerate(M_values):
            drho_rho0[i, j] = density_fluctuation(M, T, L, wavelength, NN)
            delta_n = refractive_index_fluctuation(drho_rho0[i, j], critical_density, rho_0)
            phase_shifts[i, j] = phase_shift(delta_n, L, wavelength)
            beam_deviations[i, j] = beam_deviation(delta_n, L, wavelength)
    plt.figure(figsize=(13.385, 6.25))
    #   fig.set_size_inches(13.385, 6.0)
    levels = 1000
    vmin = 0
    vmax = 0.5
    color = plt.contourf(M_values, 1000*L_values, drho_rho0, levels=levels, cmap='RdBu', vmin=vmin, vmax=vmax, extend='both')

    # # Add a colorbar
    # cbar = plt.colorbar(color)

    # # Set specific tick marks on the colorbar
    # cbar.set_ticks(np.linspace(vmin, vmax, num=5))  # Adjust num to control the number of ticks

    plt.xlabel('Mach Number (M)')
    plt.ylabel('L (mm)')
    plt.tight_layout
    # Save the plot with the size in inches and 600 DPI
    plt.savefig('D:\XSPL\Papers\Thesis\Figures'+'\density_fluctuation.tif', dpi=600)
    plt.show()
    return M_values, L_values, phase_shifts, beam_deviations

# Parameters
wavelength = 532e-9  # Laser wavelength in meters
epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
m_e = 9.11e-31  # Electron mass in kg
e = 1.6e-19  # Elementary charge in C
k_B = 1.38e-23  # Boltzmann constant, J/K
omega = 2 * np.pi * 3e8 / wavelength  # Angular frequency of the laser
m_p = 1.67e-27  # Mass of a proton in kg

n_critical = epsilon_0 * m_e * omega**2 / e**2  # Critical number density for the laser wavelength in m^-3
print(f"Critical density: {n_critical:.2e} m^-3")
# Assume initial number density of plasma (n_e0)
n_e0 = 1e25  # Example number density in m^-3

# Mach number and length ranges
mach_range = (0.1, 3.0)
L_range = (0.0001, 0.001)

temp = 10 *(e/k_B)

NN = 27 # Ion mass number for Al ions

# Plot the density fluctuation
M_values, L_values, phase_shifts, beam_deviations = plot_density_fluctuation(L_range, wavelength, mach_range, temp, n_e0, n_critical, NN)

# Plot phase shift
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(M_values, 1000*L_values, phase_shifts, levels=100, cmap='RdBu')
plt.colorbar(label='Phase Shift (radians)')
plt.xlabel('Mach Number (M)')
plt.ylabel('L (m)')
plt.title('Phase Shift vs. Mach Number and Temperature')

# Plot beam deviation
plt.subplot(1, 2, 2)
plt.contourf(M_values, 1000*L_values, beam_deviations, levels=100, cmap='plasma')
plt.colorbar(label='Beam Deviation (radians)')
plt.xlabel('Mach Number (M)')
plt.ylabel('L (m)')
plt.title('Beam Deviation vs. Mach Number and Temperature')

plt.tight_layout()
plt.show()
