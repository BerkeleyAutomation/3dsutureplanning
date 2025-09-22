import DistanceCalculator
import math
import numpy as np

class Constraints:
    def __init__(self, wound_width, centroids=None):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width # TODO Varun: this is a random #, lookup
        self.centroids = centroids

    def con2(self, t):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)   
        h = self.wound_width * (1/5)
        return [i - h for i in insert_dists] + [i - h for i in center_dists] + [i - h for i in extract_dists]
    
    def con3(self, t): # max distance b/w 2 sutures
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)   
        h = self.wound_width * 4
        return [h - i for i in insert_dists] + [h - i for i in center_dists] + [h - i for i in extract_dists]

    def con4(self, t):
        return [t[i+1] - t[i] for i in range(len(t)-1)]
    
    def con5(self, t):
        if not self.centroids:
            return []

        suture_points = [self.DistanceCalculator.wound_parametric(ti, 0) for ti in t]
        epsilon = 0.1 # in mm
        constraints = []
        for c in self.centroids:
            min_dist = min([self.euc_dist(s, c) for s in suture_points])
            constraints.append(epsilon - min_dist)
        return constraints


    def constraints(self):
        start = self.wound_points[0]
        end = self.wound_points[-1]

        # start = 0
        # end = 1 # NOTE: it should always be this way!
        return ({'type': 'ineq', 'fun': lambda t: t[0] - start}, {'type': 'ineq', 'fun': lambda t: t[-1] - end}, 
               {'type': 'ineq', 'fun': lambda t: - t[0] + start}, {'type': 'ineq', 'fun': lambda t: - t[-1] + end}, 
               {'type': 'ineq', 'fun': lambda t: self.con2(t)},
               {'type': 'ineq', 'fun': lambda t: self.con3(t)},
               {'type': 'ineq', 'fun': lambda t: t - start},
               {'type': 'ineq', 'fun': lambda t: end - t},
               {'type':'ineq', 'fun': lambda t: self.con5(t)}
                #{'type': 'ineq', 'fun': lambda t: self.con4(t)},
               )
    
    def euc_dist(self, x, y):
        pixel_dist = math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
        mm_dist = pixel_dist * 1.0 / self.DistanceCalculator.pixels_per_mm
        return mm_dist
    
    def get_closest_suture(self, t):
        if not self.centroids:
            return []

        suture_points = [self.DistanceCalculator.wound_parametric(ti, 0) for ti in t]
        epsilon = 0.1 # in mm
        constraints = []
        for c in self.centroids:
            min_dist = np.argmin([self.euc_dist(s, c) for s in suture_points])
            constraints.append(min_dist)
        return constraints