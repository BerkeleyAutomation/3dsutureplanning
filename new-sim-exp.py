import matplotlib.pyplot as plt 
import numpy as np
from MeshIngestor import MeshIngestor
from scipy.spatial import cKDTree
import networkx as nx
import scipy.interpolate as inter
from scipy.interpolate import splprep, splev
from Optimizer3d import Optimizer3d



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

    mesh.is_mesh = True

    spline_x_start, spline_x_end = -0.1, 0.3

    # make sim spline
    spline = make_spline_pts(spline_x_start, spline_x_end, surface_function, 50)
    x_line, y_line, z_line = spline
    # if viz:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), c=z_grid.flatten(), cmap='viridis', marker='o')
    #     ax.plot(x_line, y_line, z_line, color='r', marker='o', label='Line')
    #     plt.show()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def width_function(t):
        # return 0.03 * (1 - 2 * np.abs(t - 0.5))
        return 0.02*(1-2 *np.abs(t-0.5))+0.01

    # Compute normal vectors for the spline
    def compute_normals(x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.sqrt(dx**2 + dy**2)
        
        # Normalized tangent vectors
        tx, ty = dx / norm, dy / norm

        # Rotate 90 degrees to get normals
        nx, ny = -ty, tx
        return nx, ny

    # Generate base spline
    spline_x_start, spline_x_end = -0.1, 0.3
    x_spline, y_spline, z_spline = make_spline_pts(spline_x_start, spline_x_end, surface_function, 50)

    # Compute normal vectors
    nx, ny = compute_normals(x_spline, y_spline)

    # Generate left and right offsets
    t_values = np.linspace(0, 1, len(x_spline))
    w = width_function(t_values)  # Compute width at each point
    left_x, left_y = x_spline + w * nx, y_spline + w * ny
    right_x, right_y = x_spline - w * nx, y_spline - w * ny

    # Compute Z values for left and right splines
    left_z = np.array([surface_function(left_x[i], left_y[i]) for i in range(len(left_x))])
    right_z = np.array([surface_function(right_x[i], right_y[i]) for i in range(len(right_x))])

    # Create a surface fill using triangles
    faces = []
    for i in range(len(left_x) - 1):
        faces.append([
            [left_x[i], left_y[i], left_z[i]],
            [right_x[i], right_y[i], right_z[i]],
            [left_x[i + 1], left_y[i + 1], left_z[i + 1]]
        ])
        faces.append([
            [right_x[i], right_y[i], right_z[i]],
            [right_x[i + 1], right_y[i + 1], right_z[i + 1]],
            [left_x[i + 1], left_y[i + 1], left_z[i + 1]]
        ])

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original surface
    ax.scatter(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), c=z_grid.flatten(), cmap='viridis', marker='o', alpha=0.3)

    # Plot center spline
    # ax.plot(x_spline, y_spline, z_spline, color='r', label="Center Spline", marker='o')

    # Plot width-adjusted splines
    ax.plot(left_x, left_y, left_z, color='r', label="Left Boundary", marker='o')
    ax.plot(right_x, right_y, right_z, color='r', label="Right Boundary", marker='o')

    # Fill in the surface
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, facecolor='red', edgecolor='none'))


    border_pts_3d = np.vstack((
        np.column_stack((left_x, left_y, left_z)),  # Left boundary
        np.column_stack((right_x, right_y, right_z))  # Right boundary
    ))

    # define t based on cumulative dists
    spline3d = [inter.UnivariateSpline(t_values, x_spline, s=0), inter.UnivariateSpline(t_values, y_spline, s=0), inter.UnivariateSpline(t_values, z_spline, s=0)]

    granularity = 99

    x_pts = [spline3d[0](t/granularity) for t in range(granularity+1)]
    y_pts = [spline3d[1](t/granularity) for t in range(granularity+1)]
    z_pts = [spline3d[2](t/granularity) for t in range(granularity+1)]

    ax.plot(x_pts, y_pts, z_pts, color='b', label="Center Spline", marker='o')

    ax.legend()
    plt.show()

    derivative_x, derivative_y, derivative_z = spline3d[0].derivative(), spline3d[1].derivative(), spline3d[2].derivative()
    derivative_x2, derivative_y2, derivative_z2 = spline3d[0].derivative(2), spline3d[1].derivative(2), spline3d[2].derivative(2)
    

    curvature_arr = []
    for i in range(granularity+1):
        t = i / granularity
        r_prime = np.array([derivative_x(t), derivative_y(t), derivative_z(t)])
        r_double_prime = np.array([derivative_x2(t), derivative_y2(t), derivative_z2(t)])
        curvature = np.linalg.norm(np.cross(r_prime, r_double_prime)) / np.linalg.norm(r_prime)**3
        # print(f"AT MIDPOINT T {midpt_t} the CURVATURE IS", curvature)
        curvature_arr.append(curvature)
    
    print('CURVATURE', curvature_arr)
    print("MIN", np.min(curvature_arr))
    print("MAX", np.max(curvature_arr))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Spline curvature")
    print("max curve", max(curvature_arr))
    p = ax.scatter3D(x_pts, y_pts, z_pts, c=curvature_arr)
    fig.colorbar(p)
    plt.show()

    def sigmoid(x, L, k, x0):
        """
        Sigmoid function with parameters to control its shape.
        L: the curve's maximum value
        k: the logistic growth rate or steepness of the curve
        x0: the x-value of the sigmoid's midpoint
        """
        return L / (1 + np.exp(-k * (x - x0)))

    curvature_arr = np.array(curvature_arr)
    scaled_curvature = curvature_arr / max(curvature_arr)
    L = 1/0.5 - 0.77  # The range of the spacing values
    # more curve means 1/0.5 ellipse whereas less curve means greater
    k = 10  # The steepness of the curve
    x0 = 0.5  # The midpoint of the sigmoid

    # Calculate the spacing using the sigmoid function
    spacing = 0.77 + sigmoid(scaled_curvature, L, k, x0)
    print('SPACING', spacing)
    # get mesh from the surrounding points

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Sigmoid eccentricity")
    p = ax.scatter3D(x_pts, y_pts, z_pts, c=spacing)
    fig.colorbar(p)
    plt.show()


    #hyperparameters
    gamma = 0.005 # ideal distance between each suture
    c_ideal = 0 # variance from ideal
    c_var = 10 # variance between center points distances
    c_shear = 0 # shear loss
    c_closure = 0 # closure loss

    hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

    force_model_parameters = {'ellipse_ecc': 1/0.5, 'force_decay': 0.5/0.005, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

    optim3d = Optimizer3d(mesh, spline3d, 0.005, hyperparams, force_model_parameters, spline3d, spacing, None, border_pts_3d)

    spline_length = optim3d.calculate_spline_length(spline3d, mesh)
    start_range = 5
    end_range = 7

    print("range:", start_range, end_range)

    equally_spaced_losses = {}
    post_algorithm_losses = {}


    best_baseline_loss = 1e8
    best_baseline_placement = None

    best_opt_loss = 1e8
    best_opt_insertion = None
    best_opt_extraction = None
    best_opt_center = None

    final_closure = None
    final_shear = None

    for num_sutures in range(start_range, end_range + 1):
        print("Num sutures:", num_sutures)

        center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, spline3d, num_sutures=num_sutures)
        #print("Normal vector", normal_vectors)
        optim3d.plot_mesh_path_and_spline()
        equally_spaced_losses[num_sutures] = optim3d.optimize(eval=True)
        print('Initial loss', equally_spaced_losses[num_sutures]["curr_loss"])
        optim3d.optimize(eval=False)
    
        # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=True, results_pth=baseline_pth)
        # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=viz, results_pth=opt_pth)

        post_algorithm_losses[num_sutures] = optim3d.optimize(eval=True)
        print('After loss', post_algorithm_losses[num_sutures]["curr_loss"])
        
        if post_algorithm_losses[num_sutures]["curr_loss"] < best_opt_loss:
            # print("Num sutures", num_sutures, "best loss so far")
            best_opt_loss = post_algorithm_losses[num_sutures]["curr_loss"]
            print("BEST LOSS", best_opt_loss)
            best_opt_insertion = optim3d.insertion_pts
            best_opt_extraction = optim3d.extraction_pts
            best_opt_center = optim3d.center_pts
            baseline_insertion = insertion_pts
            baseline_extraction = extraction_pts
            baseline_center = center_pts
            _, _, final_closure, _, _, _, _ = optim3d.compute_closure_shear_loss(granularity=100)


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("CLOSURE FORCES FINAL")
    p = ax.scatter3D(x_pts, y_pts, z_pts, c=final_closure)
    fig.colorbar(p)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("SHEAR FORCES FINAL")
    p = ax.scatter3D(x_pts, y_pts, z_pts, c=final_shear)
    fig.colorbar(p)
    plt.show()