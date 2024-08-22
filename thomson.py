import plasmapy as pp
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from plasmapy.diagnostics import thomson


# Calculate the Thomson scattering power for an incoming em wave with wavelength lambda
# The probe wavelength can in theory be anything, but in practice integer frequency multiples of the Nd:YAG wavelength
# 1064 nm are used (532 corresponds to a frequency-doubled probe beam from such a laser).
probe_wavelength = 0.155 * u.nm
probe_wavelength = 10*0.155 * u.nm
probe_wavelength = 532 * u.nm

# Array of wavelengths over which to calculate the spectral distribution
range = 0.0001
wavelengths = (
    np.arange(probe_wavelength.value - range*probe_wavelength.value, probe_wavelength.value + range*probe_wavelength.value, range/1e3) * u.nm
)

# The scattering geometry is defined by unit vectors for the orientation of the probe laser beam (probe_n) and
# the path from the scattering volume (where the measurement is made) to the detector (scatter_n).
# These can be setup for any experimental geometry.
probe_vec = np.array([0, 0, 1])
scattering_angle = np.deg2rad(90)
scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

ne = 2e29 * u.m**-3
T_e = 1500 * u.eV
T_i = 1000* u.eV

ne = 5e25 * u.m**-3
T_e = 1 * u.eV
T_i = 1* u.eV

alpha, Skw = thomson.spectral_density(
    wavelengths,
    probe_wavelength,
    ne,
    T_e=T_e,
    T_i=T_i,
    probe_vec=probe_vec,
    scatter_vec=scatter_vec,
    electron_vel=np.array([[0, 0, 0]]) * u.km / u.s,
    ion_vel=np.array([[0, 0, 0]]) * u.km / u.s,
    ions=['N 2+'],
    # ions=['Al 4+'],
)
print("alpha: ", alpha)
fig, ax = plt.subplots()
ax.plot(wavelengths, Skw, lw=2)
ax.set_xlim(probe_wavelength.value - range*probe_wavelength.value, probe_wavelength.value + range*probe_wavelength.value)
# ax.set_ylim(0, 1e-13)
ax.set_xlabel(r"$\lambda$ (nm)")
ax.set_ylabel("S(k,w)")
ax.set_title("Thomson Scattering Spectral Density");
plt.show()