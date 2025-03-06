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

def plot_variable_in_region(time, run_dir, variable_name, x_bounds, y_bounds, cmap='terrain', clim=None, to_save=False, save_dir=None):
    """
    Plots a selected variable from the combined grid at a given timestep within specified x and y bounds.

    Parameters:
    - time: Time step to load.
    - run_dir: Directory containing the simulation tiles.
    - variable_name: Name of the variable to plot.
    - x_bounds: Tuple (x_min, x_max) specifying the region in the x-axis.
    - y_bounds: Tuple (y_min, y_max) specifying the region in the y-axis.
    - cmap: Colormap for the plot.
    - clim: Color limits for the plot (optional).
    """
    # Combine tiles into a single grid for the given timestep
    combined_grid = combine_tiles(time, run_dir)
    
    # Convert to structured grid
    smesh = unstructured_to_structured(combined_grid, variable_name)
    
    # Extract mesh grid coordinates
    x_coords = np.unique(smesh.points[:, 0])
    y_coords = np.unique(smesh.points[:, 1])
    
    # Find indices for the specified bounds
    x_indices = np.where((x_coords >= x_bounds[0]) & (x_coords <= x_bounds[1]))[0]
    y_indices = np.where((y_coords >= y_bounds[0]) & (y_coords <= y_bounds[1]))[0]
    
    # Ensure variable exists
    if variable_name not in smesh.point_data:
        raise ValueError(f"Variable '{variable_name}' not found in the dataset. Available variables: {list(smesh.point_data.keys())}")
    
    # Extract variable data and reshape to match grid dimensions
    nx, ny, _ = smesh.dimensions
    data = smesh.point_data[variable_name].reshape((nx, ny), order='F')
    
    # Select the subset of data
    selected_data = data[np.ix_(x_indices, y_indices)]
    selected_x = x_coords[x_indices]
    selected_y = y_coords[y_indices]
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(selected_x, selected_y, selected_data.T, shading='auto', cmap=cmap)
    plt.gca().set_aspect('equal')  # Automatically adjusts to data
    if clim:
        plt.clim(*clim)
    
    if to_save:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{variable_name.replace(" ", "_")}_time_{time}.png'), dpi=600)  
    plt.colorbar(label=variable_name)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'{variable_name} in Selected Region (Time {time})')
    plt.show()

def plot_avg_variable_over_y(time, run_dir, variable_name, x_bounds, y_range, to_plot=True, ax=None, bounds=None, use_right_axis=False, plot_style={}, to_save=False, save_dir=None):
  """
  Computes the average of a variable over a range of y values and plots it as a function of x.

  Parameters:
  - time: Time step to load.
  - run_dir: Directory containing the simulation tiles.
  - variable_name: Name of the variable to average.
  - x_bounds: Tuple (x_min, x_max) specifying the region in the x-axis.
  - y_range: Tuple (y_min, y_max) specifying the y range to average over.
  - ax: Matplotlib axis object to plot on (optional).
  - use_right_axis: If True, plots on the right-side y-axis.
  - plot_style: Dictionary of style settings (e.g., linewidth, color, linestyle).
  """
  # Combine tiles into a single grid for the given timestep
  combined_grid = combine_tiles(time, run_dir)
  
  # Convert to structured grid
  smesh = unstructured_to_structured(combined_grid, variable_name)
  
  # Extract mesh grid coordinates
  x_coords = np.unique(smesh.points[:, 0])
  y_coords = np.unique(smesh.points[:, 1])
  
  # Find indices for the specified bounds
  x_indices = np.where((x_coords >= x_bounds[0]) & (x_coords <= x_bounds[1]))[0]
  y_indices = np.where((y_coords >= y_range[0]) & (y_coords <= y_range[1]))[0]
  
  # Ensure variable exists
  if variable_name not in smesh.point_data:
      raise ValueError(f"Variable '{variable_name}' not found in the dataset. Available variables: {list(smesh.point_data.keys())}")
  
  # Extract variable data and reshape to match grid dimensions
  nx, ny, _ = smesh.dimensions
  data = smesh.point_data[variable_name].reshape((nx, ny), order='F')
  
  # Compute the average over the selected y range
  test = data[x_indices[0]:x_indices[-1], y_indices]
  avg_data = np.mean(data[x_indices[0]:x_indices[-1]+1, y_indices], axis=1)
  
  # Select corresponding x values
  selected_x = x_coords[x_indices]
  
  # Plot
  if ax is None:
      fig, ax = plt.subplots(figsize=(13.385, 13.385))
  elif use_right_axis:
        ax = ax.twinx()  # Create secondary y-axis on the right
  ax.plot(selected_x, avg_data, label=f'Average {variable_name}', **plot_style)
  if bounds is not None:
      ax.set_ylim(bounds[0], bounds[1])  # Set y-axis limits from 0 to 6
  if to_save:
      plt.tight_layout()
      plt.savefig(os.path.join(save_dir, f'avg_{variable_name.replace(" ", "_")}_time_{time}.png'), dpi=600)
  plt.xlabel('X Position')
  plt.ylabel(f'Avg {variable_name}')
  plt.title(f'Average {variable_name} Over Y-Range {y_range} (Time {time})')
  # plt.legend()
  # plt.grid()
  if to_plot:
    plt.show()
  return ax

# Run Directory
# Check which file system is being used
osx = False
if os.name == 'posix':
    osx = True


# If mac osx #
if osx:
    datadir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Data/'
    savedir = '/Volumes/T9/XSPL/PERSEUS/xpinch/Bluehive/Plots/'
else:
    drive_letter = 'D:'

    data_path_on_external_drive = 'XSPL/Projects/DopedRod/Analysis/Data/PERSEUS/' 
    plot_path_on_external_drive = 'XSPL/Projects/DopedRod/Analysis/Plots/'  

    datadir = drive_letter + '\\' + data_path_on_external_drive       
    savedir = drive_letter+'\\'+plot_path_on_external_drive


run = 'run1'
run_path = datadir+run+'/data/'

time_analyze = 70
time = find_max_time(run_path)

# Combine the tiles into a single grid
mesh = combine_tiles(time_analyze, run_path)

# Get an array of x, y, z coordinates from the grid
r = mesh.points[:, 0]
z = mesh.points[:, 1]
dr = r[1]-r[0]
dz = dr

# Plot the variable in a specified region
plot_variable_in_region(time=85, run_dir=run_path, variable_name='Log Ion Density', x_bounds=(-1, 1), y_bounds=(-1, 1), to_save=True, save_dir=savedir)

style1 = {'linewidth': 4, 'color': 'k', 'linestyle': '--'}
style2 = {'linewidth': 4, 'color': 'xkcd:sky blue', 'linestyle': '--'}
style3 = {'linewidth': 4, 'color': 'xkcd:sky blue', 'linestyle': '-'}
style4 = {'linewidth': 4, 'color': 'xkcd:light red', 'linestyle': '--'}
style5 = {'linewidth': 4, 'color': 'xkcd:light red', 'linestyle': '-'}

ax = plot_avg_variable_over_y(time=0, run_dir=run_path, variable_name='Log Ion Density', x_bounds=(0, 1e-3), y_range=(-10e-6, 10e-6), bounds=(20, 30), to_plot=False, use_right_axis=False, plot_style=style1, to_save=False, save_dir=savedir)
ax = plot_avg_variable_over_y(time=65, run_dir=run_path, variable_name='Log Ion Density', x_bounds=(0, 1e-3), y_range=(-10e-6, 10e-6), bounds=(20, 30), ax=ax, use_right_axis=False, to_plot=False, plot_style=style2, to_save=False, save_dir=savedir)
ax = plot_avg_variable_over_y(time=65, run_dir=run_path, variable_name='Ion Temperature', x_bounds=(0, 250e-6), y_range=(-10e-6, 10e-6), bounds=(0, 2), ax=ax, to_plot=False, use_right_axis=True, plot_style=style4, to_save=False, save_dir=savedir)
ax = plot_avg_variable_over_y(time=85, run_dir=run_path, variable_name='Log Ion Density', x_bounds=(0, 1e-3), y_range=(-10e-6, 10e-6), bounds=(20, 30), ax=ax, to_plot=False, plot_style=style3, to_save=False, save_dir=savedir)
ax = plot_avg_variable_over_y(time=85, run_dir=run_path, variable_name='Ion Temperature', x_bounds=(0, 250e-6), y_range=(-10e-6, 10e-6), bounds=(0, 2), ax=ax, to_plot=True, use_right_axis=True, plot_style=style5, to_save=True, save_dir=savedir)

# Convert to StructuredGrid
smesh = unstructured_to_structured(mesh, variable_name='Log Ion Density')

# # Plotting
r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True)
r, z, dr, dz = plot_mesh_with_time_slider(smesh, 'Magnetic Field', cmap='terrain', clim=[0,250], to_plot=True)

# Plot the structured grid and show the axis and labels    
# plotter1 = plot_mesh_with_scalar(smesh, 'Log Ion Density', cmap='terrain', clim=[20, 30], to_plot=True, plotter_loc=[0, 0], columns=1, rows=1)
# plotter2 = plot_mesh_with_scalar(smesh, 'Magnetic Field', cmap='terrain', clim=[0, 100], to_plot=True, plotter_loc=[0, 0])

# Get the cell data for the electron density from the structured grid
data = smesh.point_data['Ion Density']

# Grid dimensions (assuming you know these or retrieve them from the grid)
nx, nz, nu = smesh.dimensions