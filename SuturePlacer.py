import random
import DistanceCalculator
import RewardFunction
import Constraints
import scipy.optimize as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import math
from scipy.interpolate import splprep, splev
from scipy.signal import argrelextrema

class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel, sampled_spline_pts):
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        self.sampled_spline_pts = sampled_spline_pts
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
        self.RewardFunction = RewardFunction.RewardFunction(wound_width, self)
        self.Constraints = Constraints.Constraints(wound_width)
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        self.b_insert_pts = []
        self.b_center_pts = []
        self.b_extract_pts = []
        self.b_loss = float('inf')

        self.c_lossMin = 0
        self.c_lossIdeal = 1
        self.c_lossVarCenter = 12
        self.c_lossVarInsExt = 6
        self.c_lossClosure = 15
        self.c_lossShear = 5

    # high curvature segmentation
    def compute_curvature_points(self, spline_pts, min_spacing=10, dense_factor=8, min_curvature=0.005):
        pts = np.asarray(spline_pts)
        n_pts = len(pts)
        if n_pts < 4:
            return np.unique(np.array([0, n_pts-1], dtype=int))

        x = pts[:, 0]
        y = pts[:, 1]

        # fit b-spline (cubic, or use quadratic)
        try:
            tck, u = splprep([x, y], s=0, k=3)
        except Exception:
            tck, u = splprep([x, y], s=0, k=2)

        n_dense = max(int(n_pts * dense_factor), 1000)
        u_dense = np.linspace(0, 1, n_dense)
        x_dense, y_dense = splev(u_dense, tck, der=0)
        dx, dy = splev(u_dense, tck, der=1)
        ddx, ddy = splev(u_dense, tck, der=2)

        dx = np.array(dx)
        dy = np.array(dy)
        ddx = np.array(ddx)
        ddy = np.array(ddy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)

        # find local maxima
        local_max_idx = argrelextrema(curvature, np.greater)[0]

        # filter by the minimum curvature
        high_curv_idx_dense = [i for i in local_max_idx if curvature[i] >= min_curvature]

        # always include endpoints
        high_curv_idx_dense = [0] + high_curv_idx_dense + [n_dense-1]

        # map dense indices back to original sampled points
        dense_pts = np.vstack([x_dense, y_dense]).T
        selected_indices = []
        for idx in high_curv_idx_dense:
            pt = dense_pts[idx]
            closest_orig_idx = int(np.argmin(np.linalg.norm(pts - pt, axis=1)))
            selected_indices.append(closest_orig_idx)

        # enforce a minimum spacing along arc length
        selected_indices = sorted(set(selected_indices))
        final_indices = [selected_indices[0]]
        for idx in selected_indices[1:]:
            if np.linalg.norm(pts[idx] - pts[final_indices[-1]]) >= min_spacing:
                final_indices.append(idx)

        return np.array(final_indices, dtype=int)

    def initial_sutures_from_curvature(self, min_spacing=10, dense_factor=8, min_curvature=1e-6, scale_factor=0.05):
        high_curv_idx = self.compute_curvature_points(
            self.sampled_spline_pts,
            min_spacing=min_spacing,
            dense_factor=dense_factor,
            min_curvature=min_curvature
        )
        suture_pts = self.segment_along_curve(self.sampled_spline_pts, high_curv_idx, scale_factor=scale_factor)
        return np.array(suture_pts)

    def segment_along_curve(self, spline_pts, high_curv_idx, scale_factor=0.05, min_suture_spacing=2.0):
        pts = np.array(spline_pts)
        if len(high_curv_idx) == 0:
            return []

        high_curv_idx = np.sort(np.array(high_curv_idx, dtype=int))
        suture_pts = []

        for i in range(len(high_curv_idx) - 1):
            start_idx = high_curv_idx[i]
            end_idx = high_curv_idx[i + 1]
            if end_idx <= start_idx:
                continue

            segment_pts = pts[start_idx:end_idx + 1]
            distances = np.linalg.norm(np.diff(segment_pts, axis=0), axis=1)
            total_dist = np.sum(distances)

            if total_dist < min_suture_spacing:
                continue

            # more sutures for higher curvature segments
            seg_curv = np.mean(np.abs(np.diff(segment_pts, axis=0)), axis=0).sum()
            num_between = max(int(total_dist * scale_factor * (1 + seg_curv)), 1)

            cum_dist = np.insert(np.cumsum(distances), 0, 0)
            sample_dist = np.linspace(0, total_dist, num_between + 2)[:-1]

            interp_pts = []
            for d in sample_dist:
                idx = np.searchsorted(cum_dist, d) - 1
                idx = max(0, min(idx, len(segment_pts) - 2))
                denom = cum_dist[idx + 1] - cum_dist[idx]
                t = 0.0 if denom == 0 else (d - cum_dist[idx]) / denom
                new_pt = (segment_pts[idx] * (1 - t) + segment_pts[idx + 1] * t)
                interp_pts.append(tuple(new_pt))

            suture_pts.extend(interp_pts)

        # include last high-curvature point
        suture_pts.append(tuple(pts[high_curv_idx[-1]]))
        return suture_pts

    def optimize(self, wound_points, optFrame):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = \
            self.DistanceCalculator.calculate_distances(wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        self.Constraints.wound_points = wound_points

        def jac(t):
            return optim.approx_fprime(t, final_loss)

        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = \
                self.DistanceCalculator.calculate_distances(t)
            self.RewardFunction.wound_points = t
            self.RewardFunction.suture_points = list(zip(insert_pts, center_pts, extract_pts))
            return self.RewardFunction.final_loss(
                c_lossMin=self.c_lossMin,
                c_lossIdeal=self.c_lossIdeal,
                c_lossVarCenter=self.c_lossVarCenter,
                c_lossVarInsExt=self.c_lossVarInsExt,
                c_lossClosure=self.c_lossClosure,
                c_lossShear=self.c_lossShear
            )

        self.progress += self.progress_incre
        optFrame.update_progress(self.progress)
        result = optim.minimize(final_loss, wound_points,
                                constraints=self.Constraints.constraints(),
                                options={"maxiter": 200},
                                method='SLSQP',
                                tol=1e-2,
                                jac=jac)

        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = \
            self.DistanceCalculator.calculate_distances(result.x)

        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts
        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x

    def place_sutures(self, _optFrame, save_figs=False):
        num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1) / 4)
        print("NUM SUTURES INITIAL:", num_sutures_initial)
        start_range = max(2, int(num_sutures_initial))
        end_range = int(2.2 * num_sutures_initial)

        _optFrame.set_suture_range(start_range, end_range)
        _optFrame.set_distance_calculator(self.DistanceCalculator)

        d = {}
        points_dict = {}
        self.progress_incre = (1 / (end_range - start_range + 1)) / 10
        _optFrame.start_range = start_range
        _optFrame.end_range = end_range

        for num_sutures in range(start_range, end_range):
            print('TESTING NUM SUTURES:', num_sutures)
            _optFrame.update_cur_sutures(num_sutures)
            self.progress = (num_sutures - start_range) / (end_range - start_range + 1)

            d[num_sutures] = {}
            suture_pts = self.initial_sutures_from_curvature(min_spacing=10, dense_factor=8, min_curvature=1e-6, scale_factor=0.05)

            if len(suture_pts) > num_sutures:
                indices = np.linspace(0, len(suture_pts) - 1, num_sutures).astype(int)
                wound_points = suture_pts[indices]
            else:
                wound_points = suture_pts

            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = \
                self.optimize(wound_points=wound_points, optFrame=_optFrame)

            _optFrame.planned_insert_pts.append(insert_pts)
            _optFrame.planned_center_pts.append(center_pts)
            _optFrame.planned_extract_pts.append(extract_pts)

            self.insert_pts = insert_pts
            self.center_pts = center_pts
            self.extract_pts = extract_pts
            points_dict[num_sutures] = ts

        _optFrame.mark_complete()
        return self.b_insert_pts, self.b_center_pts, self.b_extract_pts


def save_dict_to_file(dic, filename):
    with open(filename, 'w') as f:
        f.write(str(dic))

def load_dict_from_file():
    with open('dict.txt', 'r') as f:
        data = f.read()
    return eval(data)

def dict_to_csv(d, filename):
    rows = []
    for k, v in d.items():
        row = {"num_sutures": k}
        row.update(v)
        rows.append(row)

    df = pd.DataFrame(rows, columns=['num_sutures', 'loss', 'closure loss', 'shear loss', 'var loss'])
    df = df.sort_values(by=['loss'])
    df.to_csv(filename + ".csv", index=False)


# import random

# import DistanceCalculator
# import RewardFunction
# import Constraints
# import scipy.optimize as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime
# import os
# import math


# class SuturePlacer:
#     def __init__(self, wound_width, mm_per_pixel, sampled_spline_pts):
#         # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
#         self.wound_width = wound_width
#         self.mm_per_pixel = mm_per_pixel
#         self.sampled_spline_pts = sampled_spline_pts
#         self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
#         self.RewardFunction = RewardFunction.RewardFunction(wound_width, self)
#         self.Constraints = Constraints.Constraints(wound_width)
#         self.Constraints.DistanceCalculator = self.DistanceCalculator

#         self.b_insert_pts = []
#         self.b_center_pts = []
#         self.b_extract_pts = []
#         self.b_loss = float('inf')

#         self.c_lossMin = 0
#         self.c_lossIdeal = 1
#         self.c_lossVarCenter = 12
#         self.c_lossVarInsExt = 6
#         self.c_lossClosure = 15
#         self.c_lossShear = 5
    
#     def initial_sutures_from_curvature(self, radius=10, scale_factor=0.05, min_curvature=0.001):
#         """
#         Compute high-curvature points and generate initial suture placement.
#         """
#         high_curv_idx = self.compute_curvature_points(self.sampled_spline_pts, radius=radius, min_curvature=min_curvature)
#         suture_pts = self.segment_along_curve(self.sampled_spline_pts, high_curv_idx, scale_factor=scale_factor)
#         return np.array(suture_pts)


#     def compute_curvature_points(self, spline_pts, radius=10, min_curvature=0.001):
#         """
#         Step 1: Find local maxima of curvature
#         Step 2: Suppress peaks within `radius` (arc length in pixels)

#         spline_pts: list of (x, y) wound points
#         radius: minimum spacing between selected peaks (in pixels)
#         min_curvature: filter out weak/noisy peaks
#         """
#         pts = np.array(spline_pts)
#         x = pts[:, 0]
#         y = pts[:, 1]

#         # compute curvature
#         dx = np.gradient(x)
#         dy = np.gradient(y)
#         ddx = np.gradient(dx)
#         ddy = np.gradient(dy)
#         curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
#         curvature = np.nan_to_num(curvature)

#         # step 1: candidate local maxima
#         candidates = []
#         for i in range(1, len(curvature) - 1):
#             if curvature[i] > curvature[i-1] and curvature[i] > curvature[i+1]:
#                 if curvature[i] >= min_curvature:
#                     candidates.append((i, curvature[i]))

#         # sort by curvature strength (strongest first)
#         candidates.sort(key=lambda x: -x[1])

#         # step 2: radius suppression (non-max suppression) 
#         selected = []
#         for idx, strength in candidates:
#             if not selected:
#                 selected.append(idx)
#             else:
#                 # distance from this candidate to already selected ones
#                 dists = [np.linalg.norm(pts[idx] - pts[j]) for j in selected]
#                 if min(dists) >= radius:
#                     selected.append(idx)

#         # always include wound endpoints
#         selected.extend([0, len(spline_pts)-1])

#         return np.unique(selected)


#     def segment_along_curve(self, spline_pts, high_curv_idx, scale_factor=0.05):
#         """
#         Generate suture points along the curve between high-curvature points.

#         spline_pts: list of (x,y)
#         high_curv_idx: list of indices
#         scale_factor: fraction of distance to determine number of points

#         Returns: list of suture points along the wound
#         """
#         pts = np.array(spline_pts)
#         suture_pts = []

#         for i in range(len(high_curv_idx) - 1):
#             start_idx = high_curv_idx[i]
#             end_idx = high_curv_idx[i + 1]

#             segment_pts = pts[start_idx:end_idx + 1]
#             distances = np.linalg.norm(np.diff(segment_pts, axis=0), axis=1)
#             cum_dist = np.insert(np.cumsum(distances), 0, 0)
#             total_dist = cum_dist[-1]

#             # determine number of sutures proportional to segment length
#             num_between = int(total_dist * scale_factor)

#             # ensure at least 1 if segment has nonzero length
#             if num_between < 1 and total_dist > 0:
#                 num_between = 1

#             sample_dist = np.linspace(0, total_dist, num_between + 2)[:-1]
#             interp_pts = []

#             for d in sample_dist:
#                 idx = np.searchsorted(cum_dist, d) - 1
#                 if idx >= len(segment_pts) - 1:
#                     idx = len(segment_pts) - 2
#                 t = (d - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx])
#                 new_pt = (segment_pts[idx] * (1 - t) + segment_pts[idx + 1] * t)
#                 interp_pts.append(tuple(new_pt))

#             suture_pts.extend(interp_pts)

#         # always include the last high-curvature point
#         suture_pts.append(tuple(pts[high_curv_idx[-1]]))
#         return suture_pts


#     def optimize(self, wound_points,optFrame):
#         insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(wound_points)
#         self.RewardFunction.insert_dists = insert_dists
#         self.RewardFunction.center_dists = center_dists
#         self.RewardFunction.extract_dists = extract_dists

#         self.Constraints.wound_points = wound_points
        
#         def jac(t):
#             return optim.approx_fprime(t, final_loss)

#         def final_loss(t):
#             self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
#             self.RewardFunction.wound_points = t
#             self.RewardFunction.suture_points = list(zip(insert_pts, center_pts, extract_pts))
#             return self.RewardFunction.final_loss(c_lossMin=self.c_lossMin, c_lossIdeal = self.c_lossIdeal, c_lossVarCenter = self.c_lossVarCenter, c_lossVarInsExt=self.c_lossVarInsExt, c_lossClosure = self.c_lossClosure, c_lossShear = self.c_lossShear)

#         # continuous progress bar
#         self.progress += self.progress_incre
#         optFrame.update_progress(self.progress)

#         self.progress += self.progress_incre
#         optFrame.after(100,optFrame.update_progress,self.progress)

#         result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol=1e-2, jac = jac)
        
#         # continuous progress bar
#         self.progress += self.progress_incre
#         optFrame.update_progress(self.progress)
        
#         insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(result.x)
        
#         # continuous progress bar
#         self.progress += self.progress_incre
#         optFrame.update_progress(self.progress)

#         self.insert_pts = insert_pts
#         self.center_pts = center_pts
#         self.extract_pts = extract_pts

#         self.progress += self.progress_incre
#         optFrame.after(100,optFrame.update_progress,self.progress)

#         result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol = 1e-2, jac = jac)
#         plt.clf()
#         #save_intermittent_plots = True
#         save_intermittent_plots = False
#         if save_intermittent_plots:
#             self.DistanceCalculator.plot(result.x, "closure", plot_type='closure', save_fig='s1/' + str(len(wound_points)) + '_closure_' + str(random.randint(0, 1000000)))
#             self.DistanceCalculator.plot(result.x, "shear", plot_type='shear', save_fig='s1/' + str(len(wound_points)) + '_shear_' + str(random.randint(0, 1000000)))

#         return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x

#     def place_sutures(self, _optFrame, save_figs=False): #save_figs=True

#         # make a folder to store info

#         # if save_figs:
#         #     if not os.path.isdir("clicking"):
#         #         os.mkdir('clicking')
#         #
#         #     now = datetime.now()
#         #     # dd/mm/YY H:M:S
#         #     dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
#         #     os.mkdir('clicking/' + dt_string)
#         #     os.mkdir('clicking/' + dt_string + '/sutures')
#         #     os.mkdir('clicking/' + dt_string + '/closure')
#         #     os.mkdir('clicking/' + dt_string + '/shear')

#         num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1)) # heuristic
#         num_sutures_initial = int(num_sutures_initial / 4) # changed suture width calculations from drawing
#         print("NUM SUTURES INITIAL:", num_sutures_initial)
        
#         # Set up suture range for progress tracking
#         start_range = max(2, int(num_sutures_initial))
#         end_range = int(2.2 * num_sutures_initial)

#         # set up optimization frame
#         _optFrame.set_suture_range(start_range,end_range)
#         _optFrame.set_distance_calculator(self.DistanceCalculator)
        
#         d = {}
#         losses = {}
#         points_dict = {}
#         self.progress_incre = (1 / (end_range - start_range + 1)) / 10
#         _optFrame.start_range = start_range
#         _optFrame.end_range = end_range
#         for num_sutures in range(start_range, end_range): # This should be (0.8 * heuristic to 1.4 * heuristic)
#             print('TESTING NUM SUTURES: ', num_sutures)
            
#             # Update progress GUI
#             _optFrame.update_cur_sutures(num_sutures)
#             self.progress = (num_sutures - start_range) / (end_range - start_range + 1)
            
#             d[num_sutures] = {}
#             heuristic = num_sutures
#             best_loss = float('inf')

#             suture_pts = self.initial_sutures_from_curvature(radius=10, scale_factor=0.05)
#             # resample or limit to `num_sutures` if too many
#             if len(suture_pts) > num_sutures:
#                 indices = np.linspace(0, len(suture_pts)-1, num_sutures).astype(int)
#                 wound_points = suture_pts[indices]
#             else:
#                 wound_points = suture_pts
            
#             insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points,optFrame=_optFrame)

#             # save all suture plans for later mapping
#             _optFrame.planned_insert_pts.append(insert_pts)
#             _optFrame.planned_center_pts.append(center_pts)
#             _optFrame.planned_extract_pts.append(extract_pts)

#             # adaptive length sutures
#             dists_to_wound = []
#             for cpt in center_pts:
#                 min_dist = 10000
#                 scale_len = 0
#                 #find point in centerline that is closest to center point, save distance to outside of wound
#                 for opt in _optFrame.parent_root.ordered_pts_dist:
#                     temp_dist = math.sqrt((opt[0]-cpt[1])**2 + (opt[1]-cpt[0])**2)
#                     if temp_dist < min_dist:
#                         min_dist = temp_dist
#                         scale_len = opt[2]
#                 dists_to_wound.append(scale_len)

#             def extend_sutures(insert_pts, extract_pts, center_pts, dists_to_wound):
#                 n_insert_pts = []
#                 n_extract_pts = []
#                 for i in range(len(center_pts)):
#                     vect = (extract_pts[i][0]-insert_pts[i][0], extract_pts[i][1]-insert_pts[i][1])
#                     scale_factor = dists_to_wound[i]*0.1
#                     if scale_factor < 1.0:
#                         scale_factor = 1.0
#                     new_pt = (insert_pts[i][0] + ((scale_factor) * vect[0]), insert_pts[i][1] + ((scale_factor) * vect[1]))
#                     n_insert_pts.append(new_pt)
#                     new_pt = (extract_pts[i][0] - ((scale_factor) * vect[0]), extract_pts[i][1] - ((scale_factor) * vect[1]))
#                     n_extract_pts.append(new_pt)
                
#                 return n_extract_pts, n_insert_pts
            
#             n_insert_pts, n_extract_pts = extend_sutures(insert_pts, extract_pts, center_pts, dists_to_wound)
#             _optFrame.planned_n_insert_pts.append(n_insert_pts)
#             _optFrame.planned_n_extract_pts.append(n_extract_pts)

#             # continuous progress bar
#             self.progress += self.progress_incre
#             _optFrame.update_progress(self.progress)

#             self.RewardFunction.insert_dists = insert_dists
#             self.RewardFunction.center_dists = center_dists
#             self.RewardFunction.extract_dists = extract_dists
#             best_loss = self.RewardFunction.hyperLoss()

#             # continuous progress bar
#             self.progress += self.progress_incre
#             _optFrame.update_progress(self.progress)
            
#             # Get individual loss components
#             closure_loss = self.RewardFunction.lossClosureForce(1, 0)
#             shear_loss = self.RewardFunction.lossClosureForce(0, 1)

#             # continuous progress bar
#             self.progress += self.progress_incre
#             _optFrame.update_progress(self.progress)

#             center_var_loss = self.RewardFunction.lossVar(1, 0)
#             ins_ext_var_loss = self.RewardFunction.lossVar(0, 1)
#             ideal_loss = self.RewardFunction.lossIdeal()

#             # continuous progress bar
#             self.progress += self.progress_incre
#             _optFrame.update_progress(self.progress)
            
#             print('loss: ', best_loss)
#             print('closure loss', closure_loss)
#             print('shear loss', shear_loss)

#             # save all losses for later plotting
#             _optFrame.total_array.append(best_loss)
#             _optFrame.closure_array.append(closure_loss)
#             _optFrame.shear_array.append(shear_loss)
            
#             # Update progress GUI with loss information
#             _optFrame.update_losses(best_loss,closure_loss,shear_loss)
#             _optFrame.update_visualization(ts, f'Suture Plan: {num_sutures} Sutures')

#             d[num_sutures]['loss'] = best_loss
#             d[num_sutures]['closure loss'] = closure_loss
#             d[num_sutures]['shear loss'] = shear_loss
#             d[num_sutures]['var loss - center'] = center_var_loss
#             d[num_sutures]['var loss - ins/ext'] = ins_ext_var_loss
#             d[num_sutures]['ideal loss'] = ideal_loss
#             b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
#             losses[best_loss] = num_sutures
    
#             self.insert_pts = b_insert_pts
#             self.center_pts = b_center_pts
#             self.extract_pts = b_extract_pts

#             if best_loss < self.b_loss:
#                 self.b_loss = best_loss
#                 self.b_insert_pts = b_insert_pts
#                 self.b_center_pts = b_center_pts
#                 self.b_extract_pts = b_extract_pts

#             points_dict[num_sutures] = b_ts

#         # Mark optimization as complete
#         _optFrame.mark_complete()
        
#         dict_to_csv(d, "clicked_losses")
#         save_dict_to_file(points_dict, "clicked_points.txt")
#         return b_insert_pts, b_center_pts, b_extract_pts


# def save_dict_to_file(dic, filename):
#     f = open(filename,'w')
#     f.write(str(dic))
#     f.close()

# def load_dict_from_file():
#     f = open('dict.txt','r')
#     data=f.read()
#     f.close()
#     return eval(data)

# def dict_to_csv(d, filename):
#     rows = []
#     for k, v in d.items():
#         row = {"num_sutures": k}
#         row.update(v)
#         rows.append(row)
    
#     df = pd.DataFrame(rows, columns=['num_sutures', 'loss', 'closure loss', 'shear loss', 'var loss'])
#     df = df.sort_values(by=['loss'])
#     df.to_csv(filename + ".csv", index=False)