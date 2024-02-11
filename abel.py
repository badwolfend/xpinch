import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

# Combine the tiles into a single grid
def combine_tiles(time, run_dir):

    # Specify time as a string with leading zeros
    time_str = f'{time:04d}'

    # List all the VTK files contained in the run directory
    vtk_files = [run_dir+vtk_file for vtk_file in os.listdir(run_dir) if vtk_file.endswith(time_str+'.vtr')]

    # Load each tile
    tiles = [pv.read(vtk_file) for vtk_file in vtk_files]

    # Combine the tiles into a single grid
    combined_grid = tiles[0].copy()  # Start with a copy of the first tile
    for tile in tiles[1:]:  # Iterate over the rest of the tiles
        combined_grid = combined_grid.merge(tile)

    return combined_grid

# Convert unstructured grid to structured grid
def unstructured_to_structured(grid, variable_name='Electron Density'):
    # Assuming the grid is orthogonal and points can be mapped directly
    # Identify unique x, y, z coordinates (assuming sorted)
    x_coords = np.unique(grid.points[:, 0])
    y_coords = np.unique(grid.points[:, 1])
    z_coords = np.unique(grid.points[:, 2])

    # Check if the product of unique counts matches the total points (a necessary condition)
    if len(x_coords) * len(y_coords) * len(z_coords) == len(grid.points):
        # Proceed with conversion
        
        # Create meshgrid (assuming points are ordered and grid-like)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Create StructuredGrid
        structured_grid = pv.StructuredGrid(X, Y, Z)
        if np.allclose(structured_grid.points, grid.points):
            structured_grid.point_data[variable_name] = grid.point_data[variable_name]
        else:
            structured_grid_with_data = structured_grid.sample(grid)
    else:
        print("Cannot directly convert to StructuredGrid: point arrangement does not form a regular grid.")
    return structured_grid_with_data

def get_mesh_grid(grid):
    # Assuming the grid is orthogonal and points can be mapped directly
    # Identify unique x, y, z coordinates (assuming sorted)
    x_coords = np.unique(grid.points[:, 0])
    y_coords = np.unique(grid.points[:, 1])
    z_coords = np.unique(grid.points[:, 2])

    X, Y, Z = None, None, None
    # Check if the product of unique counts matches the total points (a necessary condition)
    if len(x_coords) * len(y_coords) * len(z_coords) == len(grid.points):
        # Proceed with conversion
        
        # Create meshgrid (assuming points are ordered and grid-like)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    else:
        print("Cannot directly convert to StructuredGrid: point arrangement does not form a regular grid.")

    return X, Y, Z

def get_mesh_subset(mesh, nxr, nzr):

    nxrange  = np.arange(nxr[0], nxr[1])
    nzrange  = np.arange(nzr[0], nzr[1])

    # Get the cell data for the electron density from the structured grid
    data = mesh.point_data['Electron Density']

    # Grid dimensions (assuming you know these or retrieve them from the grid)
    nx, nz, nu = mesh.dimensions

    Fy = np.zeros((nzr[1]-nzr[0], nxr[1]-nxr[0]))
    for zi in nzrange:
        for xi in nxrange:
            # Extract values along x for constant y and z
            value = data[(zi * nx) + xi]
            Fy[zi-nzr[0], xi-nxr[0]] = value
    return Fy

def abel_projection(radial_data, dr):
    """
    Corrected implementation to compute the Abel projection.
    
    Parameters:
    - radial_data: 1D numpy array of the radial distribution f(r).
    - dr: The spacing between consecutive data points.
    
    Returns:
    - projection_2d: The 2D projection P(x) of the radial distribution.
    """
    N = len(radial_data)
    projection_2d = np.zeros(N)
    r = np.arange(0, N * dr, dr)  # Radial positions

    for i, x in enumerate(r):
        # Only consider r values greater than x for integration
        valid_r = r[i:]
        if len(valid_r) > 1:
            integrand = 2 * valid_r * radial_data[i:] / np.sqrt(valid_r**2 - x**2+1e-10)
            projection_2d[i] = np.trapz(integrand, valid_r)
    
    return projection_2d

# # Plot the mesh with the scalar field
def plot_mesh_with_scalar(mesh, scalar_field, cmap='terrain', clim=None):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalar_field, cmap=cmap, clim=clim)
    plotter.show_axes()
    plotter.show_bounds(grid='back', location='outer', ticks='both')
    plotter.show()

# Run Directory
data_dir = '/Users/james/Documents/Data/Bluehive/PERSEUS/'
run = 'x_150um_y_m50um_t_52_R_150um_rand'
run = 'x_150um_y_0um_t_45_R_150um_rand'
run = 'R_150um_rand_er2_2'
run = 'R_85um_rand_er_2'

# run = 'perseus_power_preempt_N_1_n_48_R_150um_iner2_nolas_rand'
# run = 'perseus_power_preempt_N_1_n_48_x_150um_y_0um_t_52_R_150um'

run_dir = data_dir+run+'/data/'
time = 86
time = 9

# Combine the tiles into a single grid
mesh = combine_tiles(time, run_dir)

# Get an array of x, y, z coordinates from the grid
r = mesh.points[:, 0]
z = mesh.points[:, 1]
dr = r[1]-r[0]
dz = dr

# Convert to StructuredGrid
smesh = unstructured_to_structured(mesh, variable_name='Log Ion Density')

# Plot the structured grid and show the axis and labels    
plot_mesh_with_scalar(smesh, 'Magnetic Field', cmap='terrain', clim=None)
# plot_mesh_with_scalar(smesh, 'Log Ion Density', cmap='terrain', clim=None)

# Get the cell data for the electron density from the structured grid
data = smesh.point_data['Ion Density']

# Grid dimensions (assuming you know these or retrieve them from the grid)
nx, nz, nu = smesh.dimensions

Fy = np.zeros((nz, nx))
for zi in range(nz):

    # Constant y and z indices
    constant_z_index = 0  # Example: constant z index
    constant_nu_index = 0  # Example: constant z index

    # Calculate the start and end indices in the 1D array for the slice
    start_index = (constant_nu_index * nu * nx) + (zi * nx)
    end_index = start_index + nx

    # Extract values along x for constant y and z
    values_along_x = data[start_index:end_index]

    projection_2d = abel_projection(values_along_x, dr)
    Fy[zi, :] = projection_2d

    # Plot the results
# Attenuation for 10 keV in aluminum
mu = 2.623E+01 # cm^2/g for 10 keV in aluminum
mu = 7.955E+00 # cm^2/g for 15 keV in aluminum
# mu = 3.441E+00 # cm^2/g for 20 keV in aluminum
# mu = 1.128E+00 # cm^2/g for 30 keV in aluminum

# Convert number density to mass density
M=26.98 #g/mol for aluminum,
NA=6.022e23 # atoms/mole.

arg = Fy*1e-4 * mu * M/NA
Ip = np.exp(-arg)

# Add this Scalar value to the grid
smesh.point_data['stopping'] = Ip.flatten()

num_photons_per_pulse = 1e12
V=60e-6
H=60e-6
focal_spot_area = V*H #(VxH) in m^2

# Get the bounds of the grid
bounds = mesh.bounds

# The bounds are in the order: [xmin, xmax, ymin, ymax, zmin, zmax]
rmin, rmax, zmin, zmax, _, _ = bounds

# Print min and max values
print(f"X Min: {rmin}, X Max: {rmax}")
print(f"Y Min: {zmin}, Y Max: {zmax}")

# Define the region of interest
scl = 5
rblow = 0
rbhigh = H/2
zblow = -V/2
zbhigh = V/2

rblow = 0
rbhigh = 200e-6
zblow = -1.5e-3
zbhigh = 1.5e3

# Get Ip for a region of interest
# Assuming the region of interest is a box defined by the following limits

# Find the 1D indices for the region of interest in the mesh
r_min_index = int(rblow/ dr)
r_max_index = int(rbhigh / dr)
z_min_index = int((zblow-(zmax-zmin)/2) / dz)
z_max_index = int((zbhigh-(zmax-zmin)/2) / dz)

# Plot and make the aspect ratio consistent with the extent
# Reflect Ip about the vertical axis and plot both
Ipleft = np.flip(Ip, axis=1)
Ipfull = np.concatenate((Ipleft, Ip), axis=1)

plt.imshow(Ipfull, cmap='gray', extent=[-rmax, rmax, zmin, zmax], aspect='equal')

# sbmesh = get_mesh_subset(smesh, [r_min_index, r_max_index], [z_min_index, z_max_index])
# plt.imshow(sbmesh, cmap='terrain', aspect='equal', extent=[10**6*rblow, 10**6*rbhigh, 10**6*zblow, 10**6*zbhigh])
# plt.imshow(Ip, cmap='gray', extent=[rmin, rmax, zmin, zmax], aspect='equal')
plt.show()
