import matplotlib.pyplot as plt 
import numpy as np
from MeshIngestor import MeshIngestor
from scipy.spatial import cKDTree
import networkx as nx

if __name__ == "__main__":
    # make mesh (no need for randomness!)
    # plot a spline through the grid
    # feed into algo!
    # let's do this on the scale of roughly 50 cm to yield normal stitches
    # I need to provide all the fields for: 
    # optim3d = Optimizer3d(mesh, left_spline, suture_width, hyperparams, force_model_parameters, left_spline_smoothed, spacing, left_image, border_pts_3d)

    width_sample_granularity = 100
    viz = True
    def surface_function(x, y):
        return 0.5*np.cos(8*x) + np.cos(6*y)
    
    def spline_x_y_function(x):
        return 6*x**2 - 0.3
    
    def make_spline_pts(x_low, x_high, surface_function, granularity=50):
        samples_x = np.linspace(x_low, x_high, granularity)
        samples_y = spline_x_y_function(samples_x)
        samples_z = np.array([surface_function(samples_x[i], samples_y[i]) for i in range(len(samples_y))])
        return samples_x, samples_y, samples_z
    
    # need to fit spline to pts


    # Generate x and y coordinates using linspace
    x_values = np.linspace(-0.5, 0.5, 50)  # 50 points from -5 to 5 for x
    y_values = np.linspace(-0.5, 0.5, 50)  # 50 points from -5 to 5 for y

    # Create a meshgrid for x and y (this gives us all combinations of x and y)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Compute z values based on the surface function
    z_grid = surface_function(x_grid, y_grid)

    # Flatten the arrays to create a list of 3D points
    points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))

    # Now 'points' is a list of 3D points (x, y, z) on the surface
    print(points)

    # add points to mesh
    mesh = MeshIngestor(None, None)

    mesh.vertex_coordinates = []

    for point in points:
        mesh.vertex_coordinates.append(point)

    # Create a KD tree
    mesh.kd_tree = cKDTree(mesh.vertex_coordinates)
    mesh.vertex_coordinates = np.array(mesh.vertex_coordinates)

    spline_x_start, spline_x_end = -0.1, 0.3

    # make sim spline
    x_line, y_line, z_line = make_spline_pts(spline_x_start, spline_x_end, surface_function, 50)

    if viz:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), c=z_grid.flatten(), cmap='viridis', marker='o')
        ax.plot(x_line, y_line, z_line, color='r', marker='o', label='Line')
        plt.show()

    # wound widths
    width_fn = lambda x: x*(1-x)*0.01

    for sample in np.linspace(0, 1, 100):
        
        # get the function value at equally spaced intervals
        # position = 

        # get the eqn of the tangent plane at that point

        # step out and get nearest pt





    

    





    
