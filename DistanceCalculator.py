import math
import matplotlib.pyplot as plt
import torch as torch
import numpy as np
import os
from scipy.interpolate import make_interp_spline

class DistanceCalculator():
    def __init__(self, suturePlacer, wound_width, mm_per_pixel):
        self.suturePlacer = suturePlacer
        self.wound_width = wound_width
        self.gradients = {}
        self.pixels_per_mm = 1 / mm_per_pixel
        pass

    def distance_along(self, a, b, num_waypoints = 100):
        '''Note: Signed distance, and a can be greater than b'''
        sign = 1
        assert(a < b)
        if a > b:
            old_a = a
            a = b
            b = old_a
            sign = -1
        steps = np.linspace(a, b, num_waypoints) # must define your num_waypoints!
        wound_points = [np.array(self.wound_parametric(s, 0)) for s in steps]
        cuml_dist = 0
        wound_vectors = [wound_points[i+1] - wound_points[i] for i in range(len(wound_points) - 1)]
        for v in wound_vectors:
            cuml_dist += math.sqrt(np.sum(v ** 2))
        return sign * cuml_dist

    def calculate_distances(self, wound_point_t):
        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:

        # gets the norm of the vector (1, gradient)
        num_pts = len(wound_point_t)

        def get_norm(x, y):
            return math.sqrt(x**2 + y**2)

        # might be worth vectorizing in the future

        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)
        
        # extract the norms of the vectors
        norms = [get_norm(wound_derivatives_x[i], wound_derivatives_y[i]) for i in range(len(wound_derivatives_x))]

        # get the normal vectors as norm = 1
        normal_vecs = [[wound_derivatives_y[i]/norms[i], -wound_derivatives_x[i]/norms[i]] for i in range(num_pts)]
        
        # make norm width wound_width
        normal_vecs = [[normal_vec[0] * self.wound_width, normal_vec[1] * self.wound_width] for normal_vec in normal_vecs]

        # add and subtract for insertion and exit
        insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]
        extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in range(num_pts)]
        center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]


        # verify works by plotting
        plt.scatter([insert_pts[i][0] for i in range(num_pts)], [insert_pts[i][1] for i in range(num_pts)], c="red")
        plt.scatter([extract_pts[i][0] for i in range(num_pts)], [extract_pts[i][1] for i in range(num_pts)], c="blue")
        plt.scatter([center_pts[i][0] for i in range(num_pts)], [center_pts[i][1] for i in range(num_pts)], c="green")
        
        def euclidean_distance(point1, point2):
            x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
            return math.sqrt((x2-x1)**2+(y2-y1)**2)

        def dist_insert(i):
            return euclidean_distance(insert_pts[i], insert_pts[i+1])

        def dist_center(i):
            return euclidean_distance(center_pts[i], center_pts[i+1])

        def dist_extract(i):
            return euclidean_distance(extract_pts[i], extract_pts[i+1])

        
        insert_distances = []
        center_distances = []
        extract_distances = []
        

        for i in range(num_pts - 1):
            insert_distances.append(dist_insert(i))
            center_distances.append(dist_center(i))
            extract_distances.append(dist_extract(i))

        # save the points
        self.suturePlacer.insert_pts = insert_pts
        self.suturePlacer.center_pts = center_pts
        self.suturePlacer.extract_pts = extract_pts

        # making insert_dist/extract dist negative if there are crossings
        for i in range(len(insert_pts)-1):
            center_orientation, insert_orientation, extract_orientation = 0, 0, 0
            if self.intersect(insert_pts[i], extract_pts[i], insert_pts[i+1], extract_pts[i+1]):
                center_orientation = self.get_orientation(center_pts[i], center_pts[i+1])
                insert_orientation = self.get_orientation(insert_pts[i], insert_pts[i+1])
                extract_orientation = self.get_orientation(extract_pts[i], extract_pts[i+1])
            if insert_orientation != center_orientation:
                insert_distances[i] = insert_distances[i]*-1
            elif extract_orientation != center_orientation:
                extract_distances[i] = extract_distances[i]*-1

        return insert_distances, center_distances, extract_distances, insert_pts, center_pts, extract_pts
    
    # check for intersections
    def on_segment(self, p1, p2, p):
        return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])
    
    def cross_product(self, p1, p2):
        return p1[0] * p2[1] - p2[0] * p1[1]
    
    def direction(self, p1, p2, p3):
        return self.cross_product((p3[0]-p1[0], p3[1] - p1[1]), (p2[0]-p1[0], p2[1] - p1[1]))
    
    def get_orientation(self, p1, p2):
        if p1[0] < p2[0]:
            return 1
        else:
            return -1
    
    # checks if line segment p1p2 and p3p4 intersect and returns -1 if True and 1 when False
    def intersect(self, p1, p2, p3, p4):
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return -1
        elif d1 == 0 and self.on_segment(p3, p4, p1):
            return -1
        elif d2 == 0 and self.on_segment(p3, p4, p2):
            return -1
        elif d3 == 0 and self.on_segment(p1, p2, p3):
            return -1
        elif d4 == 0 and self.on_segment(p1, p2, p4):
            return -1
        else:
            return 1
    

    def plot(self, wound_point_t, title_plot, plot_type, save_fig=False, save_dir=False):
        plt.clf()

        # 

        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:

        # gets the norm of the vector (1, gradient)
        num_pts = len(wound_point_t)

        def get_norm(x, y):
            return math.sqrt(x**2 + y**2)

        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)

        # extract the norms of the vectors
        norms = [get_norm(wound_derivatives_x[i], wound_derivatives_y[i]) for i in range(len(wound_derivatives_x))]

        # get the normal vectors as norm = 1
        normal_vecs = [[wound_derivatives_y[i]/norms[i], -wound_derivatives_x[i]/norms[i]] for i in range(num_pts)]
          
        # make norm width wound_width
        normal_vecs = [[normal_vec[0] * self.wound_width, normal_vec[1] * self.wound_width] for normal_vec in normal_vecs]

        # add and subtract for insertion and exit
        insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]
        extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in range(num_pts)]
        center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]


        # Returns evenly spaced numbers
        # over a specified interval.
        X_, Y_ = [], []
        for i in range(500):
            t = min(wound_point_t) + (max(wound_point_t) - min(wound_point_t))*i/500
            temp = self.wound_parametric(t, 0)
            X_.append(temp[1])
            Y_.append(-temp[0])
        
        plt.plot(X_, Y_)
        plt.scatter([insert_pts[i][1] for i in range(num_pts)], [-insert_pts[i][0] for i in range(num_pts)], c="red", s = 5)
        plt.scatter([extract_pts[i][1] for i in range(num_pts)], [-extract_pts[i][0] for i in range(num_pts)], c="blue", s = 5)
        plt.scatter([center_pts[i][1] for i in range(num_pts)], [-center_pts[i][0] for i in range(num_pts)], c="green", s = 5)

        if plot_type == 'closure' or plot_type == 'shear':
            if plot_type == 'closure':
                force_to_plot = self.suturePlacer.RewardFunction.closure_forces
            elif plot_type == 'shear':
                force_to_plot = self.suturePlacer.RewardFunction.shear_forces     

            wcp_xs = self.suturePlacer.RewardFunction.wcp_xs
            wcp_ys = self.suturePlacer.RewardFunction.wcp_ys

            ax = plt.gca()
            plt.scatter(wcp_ys, [pt * -1 for pt in wcp_xs], c=force_to_plot, cmap='viridis', marker='o', s=8)
            plt.colorbar()
        else:
            for i in range(len(insert_pts)):
                plt.plot([insert_pts[i][1], extract_pts[i][1]], [-insert_pts[i][0], -extract_pts[i][0]], color = 'k')

        plt.axis('square')
        plt.title(title_plot)
        if save_fig:
            plt.savefig(save_dir + '/' + plot_type + '/' + save_fig)
        else:
            plt.show()

    def initial_number_of_sutures(self, start, end):
        dist_along_spline = self.distance_along(start, end, 100)
        return dist_along_spline/(self.wound_width*1.5)
