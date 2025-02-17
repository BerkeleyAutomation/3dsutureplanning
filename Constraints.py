import DistanceCalculator

class Constraints:
    def __init__(self, wound_width):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width # TODO Varun: this is a random #, lookup


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

                #{'type': 'ineq', 'fun': lambda t: self.con4(t)},
               )