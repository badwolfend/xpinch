import numpy as np
import matplotlib.pyplot as plt
# Constants
k_B = 1.38e-23  # Boltzmann constant, J/K
e = 1.6e-19  # Elementary charge, C
epsilon_0 = 8.854e-12  # Vacuum permittivity, F/m
m_e = 9.11e-31  # Electron mass, kg
gamma = 5/3  # Adiabatic index for monatomic gas
def calculate_phase_shift(L, delta_n, wavelength):
    """Calculate the phase shift given distance, refractive index fluctuation, and wavelength."""
    return (2 * np.pi * delta_n * L) / wavelength

def calculate_beam_deviation(L, delta_n, l):
    """Calculate the beam deviation given distance, refractive index fluctuation, and fluctuation scale."""
    return (delta_n * L) / l

def explore_scaling_vs_density_with_delta_n(temperatures, electron_densities_fractions, L_to_wavelength, fluctuation_scale=532e-9):
    """Explore the scaling of phase shift and beam deviation vs electron densities (relative to critical density)
       for a given temperature and a single laser wavelength, calculating delta_n for each electron density."""
    
    
    # Critical electron density for the given wavelength
    omega = 2 * np.pi * 3e8 / wavelength  # Laser angular frequency
    n_critical = epsilon_0 * m_e * omega**2 / e**2  # Critical electron density

    # Temperature is constant for this plot
    T = temperatures

    # Initialize arrays to store results
    phase_shifts = []
    beam_deviations = []
    
    for relative_density in electron_densities_fractions:
        n_e = relative_density * n_critical  # Convert relative density to actual electron density

        # Calculate sound speed
        c_s = np.sqrt(gamma * k_B * T / (27 * m_e))  # Approximation for ion sound speed (assuming Al ions)
        
        # Calculate the refractive index fluctuation due to density fluctuation
        delta_n = n_e / (2 * n_critical)  # Assuming delta_n is proportional to the relative density

        # Calculate phase shift and beam deviation
        phase_shift = calculate_phase_shift(L_to_wavelength, delta_n, 1)
        beam_deviation = calculate_beam_deviation(L_to_wavelength, delta_n, 1)
        
        phase_shifts.append((relative_density, phase_shift))
        beam_deviations.append((relative_density, beam_deviation))
    
    return phase_shifts, beam_deviations

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for T in [10, 20, 30]:
  temperature = e*T/k_B  # K (constant)
  temperature_ev = T
  # phase_shifts, beam_deviations = explore_scaling_vs_density_with_delta_n([T], electron_densities_fractions, 100)
  # relative_densities, phase_shift_values = zip(*phase_shifts)
  # Define ranges of electron densities as fractions of the critical density and a constant temperature
  wavelength = 532e-9  # m (green light)
  electron_densities_fractions = np.linspace(0.001, 0.1, 10)  # fractions of critical density

  # Explore the scaling for the given wavelength and constant temperature
  phase_shifts, beam_deviations = explore_scaling_vs_density_with_delta_n(temperature, electron_densities_fractions, 100)

  # Plot phase shift
  relative_densities, phase_shift_values = zip(*phase_shifts)
  axs[0].plot(relative_densities, phase_shift_values, 'o-', label=f'T={temperature_ev:.1e} eV')

  axs[0].set_xlabel('Electron Density / Critical Density')
  axs[0].set_ylabel('Phase Shift (radians)')
  axs[0].set_title('Phase Shift vs. Electron Density (532 nm)')
  axs[0].legend(loc='upper left', fontsize='small')

  # Plot beam deviation
  relative_densities, beam_deviation_values = zip(*beam_deviations)
  axs[1].plot(relative_densities, np.array(beam_deviation_values) * (180/np.pi), 'o-', label=f'T={temperature_ev:.1e} eV')

  axs[1].set_xlabel('Electron Density / Critical Density')
  axs[1].set_ylabel('Beam Deviation (degrees)')
  axs[1].set_title('Beam Deviation vs. Electron Density (532 nm)')
  axs[1].legend(loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()
