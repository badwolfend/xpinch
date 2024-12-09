import numpy as np
import matplotlib.pyplot as plt

def compute_xray_attenuation_map(diameter, length, linear_mass_density, attenuation_coefficient, resolution):
    """
    Generate a 2D map of X-ray attenuation for a cylindrical wire.

    Parameters:
    - diameter (float): Diameter of the wire in cm.
    - length (float): Length of the wire in cm.
    - linear_mass_density (float): Linear mass density of the wire material (g/cm).
    - attenuation_coefficient (float): Attenuation coefficient for the material (cm^2/g).
    - resolution (int): Number of grid points along the diameter and length.

    Returns:
    - attenuation_map (2D numpy array): Attenuation values for each grid point.
    - x (1D numpy array): Grid points along the radial direction (cm).
    - y (1D numpy array): Grid points along the wire length (cm).
    """
    radius = diameter / 2  # Radius of the wire (cm)
    x = np.linspace(-radius, radius, resolution)  # Radial direction
    y = np.linspace(0, length, resolution)  # Length direction
    xx, yy = np.meshgrid(x, y)

    # Compute the volume density from linear mass density
    cross_sectional_area = np.pi * radius**2  # cm^2
    volume_density = linear_mass_density / cross_sectional_area  # g/cm^3

    # Compute attenuation for each grid point
    attenuation_map = np.zeros_like(xx)
    for i in range(len(x)):
        for j in range(len(y)):
            # Distance from the center of the wire
            dist_from_center = np.abs(xx[j, i])
            if dist_from_center <= radius:
                # Path length through the wire at this position
                path_length = 2 * np.sqrt(radius**2 - dist_from_center**2)
                # Column density and attenuation
                column_density = volume_density * path_length
                attenuation_map[j, i] = np.exp(-attenuation_coefficient * column_density)
            else:
                # Outside the wire
                attenuation_map[j, i] = 1.0

    return attenuation_map, x, y

# Example parameters
diameter = 0.1  # cm
length = 5.0    # cm
linear_mass_density = 19.3  # g/cm (tungsten)
attenuation_coefficient = 0.3  # cm^2/g
resolution = 200  # Grid resolution

attenuation_map, x, y = compute_xray_attenuation_map(diameter, length, linear_mass_density, attenuation_coefficient, resolution)

# Plot the 2D attenuation map
plt.figure(figsize=(8, 6))
plt.pcolormesh(x, y, attenuation_map, shading='auto', cmap='viridis')
plt.colorbar(label='X-ray Attenuation')
plt.xlabel('Radial direction (cm)')
plt.ylabel('Length of wire (cm)')
plt.title('2D Map of X-ray Attenuation')
plt.show()