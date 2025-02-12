import matplotlib.pyplot as plt 
import numpy as np
from MeshIngestor import MeshIngestor
from scipy.spatial import cKDTree
import networkx as nx
import scipy.interpolate as inter
from scipy.interpolate import splprep, splev
from Optimizer3d import Optimizer3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from EdgeDetector import line_to_spline_3d
import copy


def make_spline_pts(x_low, x_high, surface_function, granularity=50):
        samples_x = np.linspace(x_low, x_high, granularity)
        samples_y = spline_x_y_function(samples_x)
        samples_z = np.array([surface_function(samples_x[i], samples_y[i]) for i in range(len(samples_y))])
        return np.array([samples_x, samples_y, samples_z]).T

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
        return 0.01*np.cos(100*x) + 0.01*np.cos(60*y)
    
    # based on surface_function, what is the normal vector at
    # a given x, y coordinate
    def surface_normal_vector(x, y):
        return np.array([1*np.sin(100*x), 0.6*np.sin(60*y), 1])
    
    def spline_x_y_function(x):
        # return 70*x**2 - 0.025
        return 100* np.sin(x)**2
    
    def width_function(t):
        # return 0.002*(t-t**2)+0.001
        return 0.003*(0.5+0.5*np.sin(1*np.pi * t -np.pi/2))+0.001
    
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
    
    def get_pts_and_vecs(spline, d_spline, granularity):
        x_pts = np.array([spline[0](t/(granularity-1)) for t in range(granularity)])
        y_pts = np.array([spline[1](t/(granularity-1)) for t in range(granularity)])
        z_pts = np.array([spline[2](t/(granularity-1)) for t in range(granularity)])

        # evenly spaced points along spline
        pts = np.array([x_pts, y_pts, z_pts]).T
        vecs = np.zeros((len(x_pts), 3))

        # now get the appropriate vectors
        for t in range(granularity):
            normal_vec = surface_normal_vector(pts[t][0], pts[t][1])
            spline_vec = [d_spline[0](t/(granularity-1)), d_spline[1](t/(granularity-1)), d_spline[2](t/(granularity-1))]

            left_vec = np.cross(normal_vec, spline_vec)

            # normalize
            vecs[t] = left_vec / np.linalg.norm(left_vec)

        return pts, vecs

    # Generate x and y coordinates using linspace
    x_values = np.linspace(-0.05, 0.05, 100)
    y_values = np.linspace(-0.05, 0.05, 100)

    # Create a meshgrid for x and y (this gives us all combinations of x and y)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Compute z values based on the surface function
    z_grid = surface_function(x_grid, y_grid)

    # Flatten the arrays to create a list of 3D points
    points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))

    # Now 'points' is a list of 3D points (x, y, z) on the surface
    # print("points", points)

    # add points to mesh
    mesh = MeshIngestor(None, None)

    mesh.vertex_coordinates = []

    for point in points:
        mesh.vertex_coordinates.append(point)

    # Create a KD tree
    mesh.kd_tree = cKDTree(mesh.vertex_coordinates)
    mesh.vertex_coordinates = np.array(mesh.vertex_coordinates)

    # setup for mesh is done
    mesh.is_mesh = True

    spline_x_start, spline_x_end = -0.01, 0.02

    # make sim spline
    spline_pts = make_spline_pts(spline_x_start, spline_x_end, surface_function, 50)

    # fit a spline to these points
    # calculate the first and second derivative, for later
    spline = line_to_spline_3d(spline_pts, s_factor=0)
    d_spline = [spline[0].derivative(), spline[1].derivative(), spline[2].derivative()]
    d2_spline = [spline[0].derivative(2), spline[1].derivative(2), spline[2].derivative(2)]

    # number of points to sample
    granularity = 100

    # get points along the wound and vectors on the ribbon 
    # perpendicular to the spline
    even_spline_pts, even_spline_vecs = get_pts_and_vecs(spline, d_spline, granularity)

    # get the border points

    left_pts = np.zeros((granularity, 3))
    right_pts = np.zeros((granularity, 3))

    for t in range(granularity):
        current_width = width_function(t/(granularity-1))
        left_pts[t] = even_spline_pts[t] + even_spline_vecs[t]*current_width
        right_pts[t] = even_spline_pts[t] - even_spline_vecs[t]*current_width

    left_x, left_y, left_z = left_pts[:, 0], left_pts[:, 1], left_pts[:, 2]
    right_x, right_y, right_z = right_pts[:, 0], right_pts[:, 1], right_pts[:, 2]
    

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
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original surface
    ax.scatter(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), c=z_grid.flatten(), cmap='viridis', marker='o', alpha=0.1)

    # Plot center spline
    # ax.plot(x_spline, y_spline, z_spline, color='r', label="Center Spline", marker='o')

    # Plot width-adjusted splines
    # ax.plot(left_x, left_y, left_z, color='r', label="Left Boundary", marker='o')
    # ax.plot(right_x, right_y, right_z, color='g', label="Right Boundary", marker='o')

    # Fill in the surface
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, facecolor='red', edgecolor='none'))


    border_pts_3d = np.vstack((
        np.column_stack((left_x, left_y, left_z)),  # Left boundary
        np.column_stack((right_x, right_y, right_z))  # Right boundary
    ))

    # define t based on cumulative dists
    # ax.plot(even_spline_pts[:, 0], even_spline_pts[:, 1], even_spline_pts[:, 2], color='b', label="Center Spline", marker='o')

    extra_vecs = False
    if extra_vecs:
        # plot the normal vectors
        for t in range(granularity):
            normal_vec = surface_normal_vector(even_spline_pts[t][0], even_spline_pts[t][1])
            ax.quiver(even_spline_pts[t][0], even_spline_pts[t][1], even_spline_pts[t][2], normal_vec[0], normal_vec[1], normal_vec[2], length=0.05, normalize=True)
            ax.quiver(even_spline_pts[t][0], even_spline_pts[t][1], even_spline_pts[t][2], d_spline[0](t/(granularity-1)), d_spline[1](t/(granularity-1)), d_spline[2](t/(granularity-1)), length=0.05, normalize=True)
            ax.quiver(even_spline_pts[t][0], even_spline_pts[t][1], even_spline_pts[t][2], even_spline_vecs[t][0], even_spline_vecs[t][1], even_spline_vecs[t][2], length=0.05, normalize=True)
    
    ax.set_aspect('equal')

    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()


    curvature_arr = []
    for i in range(granularity):
        t = i/(granularity-1)
        r_prime = np.array([d_spline[0](t), d_spline[1](t), d_spline[2](t)])
        r_double_prime = np.array([d2_spline[0](t), d2_spline[1](t), d2_spline[2](t)])
        curvature = np.linalg.norm(np.cross(r_prime, r_double_prime)) / np.linalg.norm(r_prime)**3
        # print(f"AT MIDPOINT T {midpt_t} the CURVATURE IS", curvature)
        curvature_arr.append(curvature)
    

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Spline curvature")
    p = ax.scatter3D(even_spline_pts[:, 0], even_spline_pts[:, 1], even_spline_pts[:, 2], c=curvature_arr)
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
    L = (1/0.4) - (1/0.60)  # The range of the spacing values
    # more curve means 1/0.5 ellipse whereas less curve means greater
    k = 10  # The steepness of the curve
    x0 = 0.5  # The midpoint of the sigmoid

    # Calculate the spacing using the sigmoid function
    spacing = (1/0.60) + sigmoid(scaled_curvature, L, k, x0)
    # get mesh from the surrounding points

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Sigmoid eccentricity")
    p = ax.scatter3D(even_spline_pts[:, 0], even_spline_pts[:, 1], even_spline_pts[:, 2], c=spacing)
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

    # no need to smooth curve, so same argument (spline) for both
    optim3d = Optimizer3d(mesh, spline, 0.005, hyperparams, force_model_parameters, spline, spacing, None, border_pts_3d, faces=faces)

    spline_length = optim3d.calculate_spline_length(spline, mesh)
    # print("SPLINE LEN", spline_length)
    start_range = int(spline_length/gamma * 0.5)
    end_range = int(spline_length/gamma * 1.1)

    start_range = 9
    end_range = 10

    # print("range:", start_range, end_range)

    equally_spaced_losses = {}
    post_algorithm_losses = {}

    best_baseline_loss = 1e8
    best_opt_loss = 1e8

    for num_sutures in range(start_range, end_range + 1):
        print("Num sutures:", num_sutures)

        center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, spline, num_sutures=num_sutures)
        #print("Normal vector", normal_vectors)
        # optim3d.plot_mesh_path_and_spline()
        equally_spaced_losses[num_sutures] = optim3d.optimize(eval=True)
        print('baseline loss', equally_spaced_losses[num_sutures]["curr_loss"])

        if equally_spaced_losses[num_sutures]['curr_loss'] < best_baseline_loss:
            best_baseline_num_sutures = num_sutures
            best_baseline_optim = copy.deepcopy(optim3d)
        optim3d.optimize(eval=False)
    
        # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=True, results_pth=baseline_pth)
        # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=viz, results_pth=opt_pth)

        post_algorithm_losses[num_sutures] = optim3d.optimize(eval=True)
        print('opt loss', post_algorithm_losses[num_sutures]["curr_loss"])

        # optim3d.plot_mesh_path_and_spline()

        
        if post_algorithm_losses[num_sutures]["curr_loss"] < best_opt_loss:
            # print("Num sutures", num_sutures, "best loss so far")
            best_opt_num_sutures = num_sutures
            best_opt_loss = post_algorithm_losses[num_sutures]["curr_loss"]
            best_baseline_loss = equally_spaced_losses[num_sutures]["curr_loss"]
            # print("[opt] total loss", best_opt_loss)
            # print("[opt] closure loss", post_algorithm_losses[num_sutures]["closure_loss"])
            # print("[opt] shear loss", post_algorithm_losses[num_sutures]["shear_loss"])

            # print("[baseline] total loss", best_baseline_loss)
            # print("[baseline] closure loss", equally_spaced_losses[num_sutures]["closure_loss"])
            # print("[baseline] shear loss", equally_spaced_losses[num_sutures]["shear_loss"])
            
            best_opt_insertion = optim3d.insertion_pts
            best_opt_extraction = optim3d.extraction_pts
            best_opt_center = optim3d.center_pts
            baseline_insertion = insertion_pts
            baseline_extraction = extraction_pts
            baseline_center = center_pts
            _, _, final_closure, _, _, _, _ = optim3d.compute_closure_shear_loss(granularity=100)
            best_optim = copy.deepcopy(optim3d)

    best_optim.plot_mesh_path_and_spline()
    # best_baseline_optim.plot_mesh_path_and_spline()

    print("[opt] total loss", post_algorithm_losses[best_opt_num_sutures]["curr_loss"])
    print("[opt] closure loss", post_algorithm_losses[best_opt_num_sutures]["closure_loss"])
    print("[opt] shear loss", post_algorithm_losses[best_opt_num_sutures]["shear_loss"])

    if equally_spaced_losses[best_baseline_num_sutures]["curr_loss"] == 1:
        print("CONSTRAINTS NOT MET IN BASELINE!")
    print("[baseline] total loss", equally_spaced_losses[best_baseline_num_sutures]["curr_loss"])
    print("[baseline] closure loss", equally_spaced_losses[best_baseline_num_sutures]["closure_loss"])
    print("[baseline] shear loss", equally_spaced_losses[best_baseline_num_sutures]["shear_loss"])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("CLOSURE FORCES FINAL")
    # p = ax.scatter3D(even_spline_pts[:, 0], even_spline_pts[:, 1], even_spline_pts[:, 2], c=final_closure)
    # fig.colorbar(p)
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.title("SHEAR FORCES FINAL")
    # p = ax.scatter3D(even_spline_pts[:, 0], even_spline_pts[:, 1], even_spline_pts[:, 2], c=final_shear)
    # fig.colorbar(p)
    # plt.show()