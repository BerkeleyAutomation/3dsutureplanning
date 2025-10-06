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
    def __init__(self, wound_width, mm_per_pixel, sampled_spline_pts,
                 dense_factor=8, min_curvature=0.001, scale_factor=0.05):
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

        # Defaults for spline sampling / curvature detection
        self.dense_factor = dense_factor
        self.min_curvature = min_curvature
        self.scale_factor = scale_factor

    def compute_curvature_points(self, spline_pts):
        """
        Find anchor points for sutures using knot-based segmentation:
        - Break the B-spline into segments using internal knots
        - Pick the single point of highest curvature in each segment
        - Always include endpoints
        - Optionally filter segments below min_curvature
        """
        pts = np.asarray(spline_pts)
        n_pts = len(pts)
        if n_pts < 4:
            return np.unique(np.array([0, n_pts - 1], dtype=int))

        x, y = pts[:, 0], pts[:, 1]

        # Fit B-spline
        try:
            tck, u = splprep([x, y], s=1.0, k=3)
        except Exception:
            tck, u = splprep([x, y], s=1.0, k=2)

        # Dense sampling for curvature
        n_dense = max(n_pts * 10, 1000)
        u_dense = np.linspace(0, 1, n_dense)
        x_dense, y_dense = splev(u_dense, tck)
        dx, dy = splev(u_dense, tck, der=1)
        ddx, ddy = splev(u_dense, tck, der=2)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)

        # Use spline knots to define segments
        knots = tck[0]
        # remove endpoints (0 and 1)
        knots = knots[(knots > 0) & (knots < 1)]
        segment_edges = np.concatenate([[0], knots, [1]])

        anchor_indices = []

        for i in range(len(segment_edges) - 1):
            u_start, u_end = segment_edges[i], segment_edges[i + 1]
            mask = (u_dense >= u_start) & (u_dense <= u_end)
            if not np.any(mask):
                continue
            local_curv = curvature[mask]
            if local_curv.size == 0:
                continue
            max_idx = np.argmax(local_curv)
            dense_idx = np.where(mask)[0][0] + max_idx

            # filter by min_curvature
            if local_curv[max_idx] >= self.min_curvature:
                anchor_indices.append(dense_idx)

        # Always include endpoints
        anchor_indices = [0] + anchor_indices + [n_dense - 1]

        # Map dense indices back to original sampled points
        dense_pts = np.vstack([x_dense, y_dense]).T
        selected_indices = []
        for idx in anchor_indices:
            pt = dense_pts[idx]
            closest_orig_idx = int(np.argmin(np.linalg.norm(pts - pt, axis=1)))
            selected_indices.append(closest_orig_idx)

        return np.unique(selected_indices)
    
    def segment_along_curve(self, spline_pts, anchor_idx, scale_factor=0.05):
        """
        Place sutures between anchor points.
        Anchors = high-curvature + endpoints from compute_curvature_points().
        scale_factor controls the density of intermediate sutures per segment.
        """
        pts = np.array(spline_pts)
        anchor_idx = np.sort(np.array(anchor_idx, dtype=int))
        suture_pts = []

        for i in range(len(anchor_idx) - 1):
            start_idx, end_idx = anchor_idx[i], anchor_idx[i + 1]
            if end_idx <= start_idx:
                continue

            segment_pts = pts[start_idx:end_idx + 1]
            distances = np.linalg.norm(np.diff(segment_pts, axis=0), axis=1)
            cum_dist = np.insert(np.cumsum(distances), 0, 0)
            total_dist = cum_dist[-1]

            # number of intermediate sutures ~ segment length
            num_between = max(int(total_dist * scale_factor), 1)

            sample_dist = np.linspace(0, total_dist, num_between + 2)[:-1]
            for d in sample_dist:
                idx = np.searchsorted(cum_dist, d) - 1
                idx = max(0, min(idx, len(segment_pts) - 2))
                t = (d - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx])
                new_pt = (segment_pts[idx] * (1 - t) + segment_pts[idx + 1] * t)
                suture_pts.append(tuple(new_pt))

        # include the last anchor
        suture_pts.append(tuple(pts[anchor_idx[-1]]))
        return suture_pts

    def initial_sutures_from_curvature(self, spline_pts):
        """
        Wrapper: compute anchor points + segment sutures
        """
        anchors = self.compute_curvature_points(spline_pts)
        sutures = self.segment_along_curve(spline_pts, anchors)
        return sutures, anchors

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
            sutures, anchors = self.initial_sutures_from_curvature(self.sampled_spline_pts)

            if len(sutures) > num_sutures:
                indices = np.linspace(0, len(sutures) - 1, num_sutures).astype(int)
                wound_points = sutures[indices]
            else:
                wound_points = sutures

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
