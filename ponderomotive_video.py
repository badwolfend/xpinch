import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
e = 1.602e-19  # Electron charge (Coulombs)
m = 9.109e-31  # Electron mass (kg)
omega = 2 * np.pi * 1e15  # Angular frequency of the oscillating field (rad/s)
E0 = 1e9  # Electric field amplitude (V/m)
k = 2 * np.pi / 1e-6  # Wave number (for wavelength = 1 micron)
T = 2 * np.pi / omega  # Period of the oscillating field

# Define the spatial range
x = np.linspace(0, 2 * np.pi / k, 1000)  # Spatial domain (one wavelength)

# Time array for animation
time_steps = np.linspace(0, 4 * T, 200)  # Animate over four periods

# Initialize the time-averaged square of the electric field
E_squared_avg = np.zeros_like(x)

# Set up the figure and axis for the animation
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Initialize empty line objects for electric field, E^2 avg, and ponderomotive force
line_E, = ax1.plot([], [], label=r'$\mathbf{E}(x, t)$')
line_E2_avg, = ax2.plot([], [], label=r'$\langle \mathbf{E}^2 \rangle(x)$', color='g')
line_F, = ax3.plot([], [], label=r'$\mathbf{F}_p(x)$', color='r')

# Set up the plot titles, labels, and axis limits
ax1.set_title('Oscillating Electric Field')
ax1.set_ylabel('Electric Field (V/m)')
ax1.set_xlim(0, x[-1] * 1e6)  # Convert x to microns for better visualization
ax1.set_ylim(-E0, E0)
ax1.grid(True)
ax1.legend()

ax2.set_title('Time-Averaged Electric Field Squared')
ax2.set_ylabel(r'$\langle \mathbf{E}^2 \rangle$ (V²/m²)')
ax2.set_xlim(0, x[-1] * 1e6)
ax2.set_ylim(0, E0**2)
ax2.grid(True)
ax2.legend()

ax3.set_title('Ponderomotive Force')
ax3.set_xlabel('Position (microns)')
ax3.set_ylabel('Ponderomotive Force (N)')
ax3.set_xlim(0, x[-1] * 1e6)
# ax3.set_ylim(-1e-18, 1e-18)  # Adjust based on the force magnitude
ax3.grid(True)
ax3.legend()

# Initialization function for the animation
def init():
    line_E.set_data([], [])
    line_E2_avg.set_data([], [])
    line_F.set_data([], [])
    return line_E, line_E2_avg, line_F

# Update function for the animation
def update(t):
    # Oscillating electric field as a function of time
    E_xt = E0 * np.cos(k * x) * np.cos(omega * t)
    
    # Update the time-averaged square of the electric field
    E_squared_avg[:] += E_xt**2 / len(time_steps)
    
    # Ponderomotive force calculation based on the time-averaged E^2
    ponderomotive_force = - (e**2 / (4 * m * omega**2)) * np.gradient(E_squared_avg, x)

    # Update the electric field plot
    line_E.set_data(x * 1e6, E_xt)  # Convert x to microns for the plot
    
    # Update the time-averaged E^2 plot
    line_E2_avg.set_data(x * 1e6, E_squared_avg)
    
    # Update the ponderomotive force plot
    line_F.set_data(x * 1e6, ponderomotive_force)
    
    return line_E, line_E2_avg, line_F

# Create the animation object
ani = FuncAnimation(fig, update, frames=time_steps, init_func=init,
                    blit=True, interval=50)

# Save the animation as an mp4 video
ani.save('ponderomotive_force_averaging.mp4', writer='ffmpeg', fps=30)

# To display the animation inline if using Jupyter Notebook
plt.show()
