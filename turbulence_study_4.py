import numpy as np
import matplotlib.pyplot as plt

def refractive_index_fluctuation(drho_rho0, critical_density, rho_0):
    """Calculate the refractive index fluctuation from the density fluctuation."""
    return drho_rho0 * rho_0 / (2 * critical_density)

def phase_shift(delta_n, L, wavelength):
    """Calculate the phase shift due to the refractive index fluctuation."""
    return (2 * np.pi * delta_n * L) / wavelength

def beam_deviation(delta_n, L, wavelength):
    """Calculate the beam deviation due to the refractive index gradient."""
    return delta_n * L / wavelength

def sound_speed(temperature, gamma=5/3, ion_mass=1.67e-27):
    """Calculate the sound speed in the plasma."""
    return np.sqrt(gamma * k_B * temperature / ion_mass)

def density_fluctuation(M, temperature, L, wavelength):
    """Estimate the density fluctuation for a given Mach number and temperature."""
    c_s = sound_speed(temperature)
    delta_u = M * c_s
    epsilon = (delta_u**3) / L  # Energy dissipation rate (approximation)
    delta_u_lambda = epsilon**(1/3) * wavelength**(1/3)  # Velocity fluctuation at small scale
    drho_rho0 = delta_u_lambda / c_s  # Relative density fluctuation
    return drho_rho0

def plot_density_fluctuation(L_range, wavelength, mach_range, T, rho_0, critical_density):
  """Plot the density fluctuation drho/rho_0 on a M vs. T 2D color plot."""
  M_values = np.linspace(*mach_range, 100)
  L_values = np.linspace(*L_range, 100)
  
  drho_rho0 = np.zeros((len(L_values), len(M_values)))
  phase_shifts = np.zeros((len(L_values), len(M_values)))
  beam_deviations = np.zeros((len(L_values), len(M_values)))

  for i, L in enumerate(L_values):
      # print(f"Calculating density fluctuation for T = {T} K...")
      for j, M in enumerate(M_values):
          drho_rho0[i, j] = density_fluctuation(M, T, L, wavelength)
          delta_n = refractive_index_fluctuation(drho_rho0[i, j], critical_density, rho_0)
          phase_shifts[i, j] = phase_shift(delta_n, L, wavelength)
          beam_deviations[i, j] = beam_deviation(delta_n, L, wavelength)
  plt.figure(figsize=(10, 6))
  plt.contourf(M_values, L_values, drho_rho0, levels=100, cmap='viridis')
  plt.colorbar(label='Relative Density Fluctuation (Δρ/ρ₀)')
  plt.xlabel('Mach Number (M)')
  plt.ylabel('L (m)')
  plt.title('Relative Density Fluctuation (Δρ/ρ₀) vs. Mach Number and Temperature')
  plt.show()
  return M_values, L_values, phase_shifts, beam_deviations
# Parameters
L = 1e-3  # Integral length scale in meters
wavelength = 532e-9  # Laser wavelength in meters
e = 1.6e-19  # Elementary charge in C
k_B = 1.38e-23  # Boltzmann constant, J/K
rho_0 = 1e-6  # Approximate mass density in kg/m^3
critical_density = 1e21 * 4.48e-26  # Critical density for the laser wavelength in kg/m^3

# Mach number and length ranges
mach_range = (0.1, 5.0)
L_range = (0.01, 0.1)

temp = 10 *(e/k_B)

# Plot the density fluctuation
M_values, L_values, phase_shifts, beam_deviations = plot_density_fluctuation(L_range, wavelength, mach_range, temp, rho_0, critical_density)

# Plot phase shift
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(M_values, L_values, phase_shifts, levels=100, cmap='RdBu')
plt.colorbar(label='Phase Shift (radians)')
plt.xlabel('Mach Number (M)')
plt.ylabel('L (m)')
plt.title('Phase Shift vs. Mach Number and Temperature')

# Plot beam deviation
plt.subplot(1, 2, 2)
plt.contourf(M_values, L_values, beam_deviations, levels=100, cmap='plasma')
plt.colorbar(label='Beam Deviation (radians)')
plt.xlabel('Mach Number (M)')
plt.ylabel('L (m)')
plt.title('Beam Deviation vs. Mach Number and Temperature')

plt.tight_layout()
plt.show()
