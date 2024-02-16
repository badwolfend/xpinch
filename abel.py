import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

# Find maximum time step in the run directory
def find_max_time(run_dir):
    # List all the VTK files contained in the run directory
    vtk_files = [vtk_file for vtk_file in os.listdir(run_dir) if vtk_file.endswith('.vtr')]

    # Check if any files were found
    if not vtk_files:
        raise FileNotFoundError(f'No VTK files found in directory {run_dir}')
    
    # Extract the time step from each file name by splitting based on '.' and 't'
    time_steps = [int(vtk_file.split('.')[0].split('t')[-1]) for vtk_file in vtk_files]

    # Return the maximum time step
    return max(time_steps)

def combine_tiles(time, run_dir):

    # Specify time as a string with leading zeros
    time_str = f'{time:04d}'

    # List all the VTK files contained in the run directory
    vtk_files = [os.path.join(run_dir, vtk_file) for vtk_file in os.listdir(run_dir) if vtk_file.endswith(time_str+'.vtr')]

    # Check if any files were found
    if not vtk_files:
        raise FileNotFoundError(f'No VTK files found for time {time_str} in directory {run_dir}')
    
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

def plot_mesh_with_time_slider(mesh, scalar_field, cmap='terrain', clim=None, to_plot=True):
    # Get an array of x, y, z coordinates from the grid
    r = mesh.points[:, 0]
    z = mesh.points[:, 1]
    dr = r[1]-r[0]
    dz = dr
    time_max = find_max_time(run_path)
    print(f"Max Time: {time_max}")

    # Function to update the plot based on the slider's value (time step)
    def update_plot(value):
        time = int(value)
        print(f"Time Step: {time}" )
        # Combine the tiles into a single grid
        mesh2 = combine_tiles(time, run_path)

        # Get an array of x, y, z coordinates from the grid
        r = mesh2.points[:, 0]
        z = mesh2.points[:, 1]
        dr = r[1]-r[0]
        dz = dr

        # Update the scalar field based on the selected time step
        plotter.add_mesh(mesh2, scalars=scalar_field, cmap=cmap, clim=clim)
        plotter.render()

    # Create the plotter
    plotter = pv.Plotter()

    # Add the mesh with the scalar field
    plotter.add_mesh(mesh, scalars=scalar_field, cmap=cmap, clim=clim)

    # Show the axes
    plotter.show_axes()

    # Show the bounds
    plotter.show_bounds(grid='back', location='outer', ticks='both')

    # Add the slider to the plotter
    plotter.add_slider_widget(update_plot, rng=[0, time_max], value=0, title='Time Step')

    # Show the plotter (this will also render the plot)
    if to_plot:
        plotter.show()
    return r, z, dr, dz



# # Plot the mesh with the scalar field
def plot_mesh_with_scalar(mesh, scalar_field, cmap='terrain', clim=None, to_plot=True, plotterext=None, plotter_loc=[0, 0], columns=1, rows=1):
    if plotterext is None:
        plotter = pv.Plotter(shape=(rows, columns ))
        plotter.subplot(plotter_loc[0], plotter_loc[1])
    else:
        plotter = plotterext
        plotter.subplot(plotter_loc[0], plotter_loc[1])
    plotter.add_mesh(mesh, scalars=scalar_field, cmap=cmap, clim=clim)
    plotter.show_axes()
    plotter.show_bounds(grid='back', location='outer', ticks='both')
    if to_plot:
        plotter.show()
    return plotter

# Run Directory
# Check which file system is being used
osx = False
if os.name == 'posix':
    osx = True


# If mac osx #
if osx:
    datadir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Data/'
    # datadir =  '/Users/james/Documents/Data/Bluehive/PERSEUS/'
    savedir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Plots/'
else:
    drive_letter = 'D:'

    data_path_on_external_drive = 'XSPL/PERSEUS/xpinch/Bluehive/Data/' 
    plot_path_on_external_drive = 'XSPL/PERSEUS/xpinch/Bluehive/Plots/'  

    datadir = drive_letter + '\\' + data_path_on_external_drive       
    savedir = drive_letter+'\\'+plot_path_on_external_drive

run = 'R_150um_rand_er2_2'
# run = 'R_85um_rand_er_2'
run = 'R_85um_rand_2mm_er_2'
run = 'R_85um_rand_2mm_er_2_lowres_2'
run = 'R_85um_rand_2mm_er_2_lowres_3'
run = 'R_85um_rand_2mm_er_2_lowres_4'

run_path = datadir+run+'/data/'

time_analyze = 94
time = find_max_time(run_path)

# Combine the tiles into a single grid
mesh = combine_tiles(time_analyze, run_path)

# Get an array of x, y, z coordinates from the grid
r = mesh.points[:, 0]
z = mesh.points[:, 1]
dr = r[1]-r[0]
dz = dr

# Convert to StructuredGrid
smesh = unstructured_to_structured(mesh, variable_name='Log Ion Density')

# # Use the glyph filter to visualize the vectors
# smesh.set_active_vectors('Magnetic Field')
# vector_data = smesh.cell_arrays['Magnetic Field']
# cell_id = 0
# print(f"Vector data for cell {cell_id}: {vector_data[cell_id]}")

# glyphs = smesh.glyph(orient='Magnetic Field')


# # Plotting
# plotter = pv.Plotter()
# plotter.add_mesh(glyphs, color='blue')
# plotter.show()

r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True)
r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Magnetic Field', cmap='terrain', clim=[0,250], to_plot=True)

# Plot the structured grid and show the axis and labels    
# plotter1 = plot_mesh_with_scalar(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True, plotter_loc=[0, 0], columns=1, rows=1)
# plotter2 = plot_mesh_with_scalar(smesh, 'Magnetic Field', cmap='terrain', clim=[0, 100], to_plot=True, plotter_loc=[0, 0])

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

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(13.385, 6.0)
plt.imshow(Ipfull, cmap='RdBu', extent=[-rmax, rmax, zmin, zmax], aspect='equal', vmin=0, vmax=1)

# sbmesh = get_mesh_subset(smesh, [r_min_index, r_max_index], [z_min_index, z_max_index])
# plt.imshow(sbmesh, cmap='terrain', aspect='equal', extent=[10**6*rblow, 10**6*rbhigh, 10**6*zblow, 10**6*zbhigh])
# cbar = fig.colorbar(lc, ax=ax)
# cbar.set_label('Color mapping value')
# show colorbar
# plt.colorbar()
plt.tight_layout()
fig.savefig(savedir+run+"_"+str(time_analyze)+'_'+str(mu)+'_absorption.png', dpi=300)
plt.show()
