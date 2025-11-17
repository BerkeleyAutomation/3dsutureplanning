"""
SuturePlacer: Core optimization class for suture placement planning.

This module contains the SuturePlacer class which optimizes suture placement
along a wound centerline by minimizing closure and shear forces.
"""

import math
import numpy as np
import pandas as pd
import scipy.optimize as optim

from . import DistanceCalculator
from . import RewardFunction
from . import Constraints


class SuturePlacer:
    """
    Optimizes suture placement along a wound centerline.
    
    Uses scipy optimization to find optimal suture positions that minimize
    closure and shear forces while respecting constraints.
    
    Attributes:
        wound_width (float): Width of the wound in millimeters
        mm_per_pixel (float): Conversion factor from pixels to millimeters
        DistanceCalculator: Calculates distances between suture points
        RewardFunction: Computes loss functions for optimization
        Constraints: Defines optimization constraints
        c_lossMin, c_lossIdeal, etc.: Loss function coefficients
    """
    
    def __init__(self, wound_width, mm_per_pixel, centroids=None):
        """
        Initialize the SuturePlacer.
        
        Args:
            wound_width (float): Width of the wound in millimeters
            mm_per_pixel (float): Conversion factor from pixels to millimeters
            centroids (list, optional): High-curvature points to prioritize
        """
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        
        # Initialize component classes
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(
            self, self.wound_width, self.mm_per_pixel
        )
        self.RewardFunction = RewardFunction.RewardFunction(wound_width, self)
        self.Constraints = Constraints.Constraints(wound_width, centroids=centroids)
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        # Track best results found during optimization
        self.b_insert_pts = []
        self.b_center_pts = []
        self.b_extract_pts = []
        self.b_loss = float('inf')

        # Loss function coefficients
        self.c_lossMin = 0
        self.c_lossIdeal = 1
        self.c_lossVarCenter = 12
        self.c_lossVarInsExt = 6
        self.c_lossClosure = 15
        self.c_lossShear = 5

    def optimize(self, wound_points, optFrame):
        """
        Optimize suture placement for a given number of sutures.
        
        Uses SLSQP optimization to minimize the loss function subject to
        constraints (minimum spacing, etc.).
        
        Args:
            wound_points (np.array): Initial parametric positions along wound (0-1)
            optFrame: GUI frame for progress updates
            
        Returns:
            tuple: (insert_dists, center_dists, extract_dists, 
                   insert_pts, center_pts, extract_pts, optimized_points)
        """
        # Calculate initial distances
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = \
            self.DistanceCalculator.calculate_distances(wound_points)
        
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        self.Constraints.wound_points = wound_points
        
        def jac(t):
            """Jacobian approximation for optimization."""
            return optim.approx_fprime(t, final_loss, epsilon=1e-6)

        def final_loss(t):
            """Loss function to minimize."""
            # Recalculate distances for new point positions
            (self.RewardFunction.insert_dists, 
             self.RewardFunction.center_dists, 
             self.RewardFunction.extract_dists, 
             insert_pts, center_pts, extract_pts) = \
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

        # Update progress
        self.progress += self.progress_incre
        optFrame.update_progress(self.progress)
        optFrame.after(100, optFrame.update_progress, self.progress)

        # Run optimization
        result = optim.minimize(
            final_loss, 
            wound_points, 
            constraints=self.Constraints.constraints(),
            options={"maxiter": 200},
            method='SLSQP',
            tol=1e-2,
            jac=jac
        )
        
        # Update progress
        self.progress += self.progress_incre
        optFrame.update_progress(self.progress)
        
        # Calculate final distances from optimized points
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = \
            self.DistanceCalculator.calculate_distances(result.x)
        
        self.progress += self.progress_incre
        optFrame.update_progress(self.progress)

        # Store results
        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts

        self.progress += self.progress_incre
        optFrame.after(100, optFrame.update_progress, self.progress)

        # Second optimization pass for refinement
        result = optim.minimize(
            final_loss, 
            wound_points, 
            constraints=self.Constraints.constraints(),
            options={"maxiter": 200},
            method='SLSQP',
            tol=1e-2,
            jac=jac
        )

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x

    def place_sutures(self, _optFrame, save_figs=False):
        """
        Main optimization loop: test different numbers of sutures and find optimal.
        
        Tests a range of suture counts and finds the configuration with minimum loss.
        Updates GUI progress and stores results for visualization.
        
        Args:
            _optFrame: GUI frame for progress updates and result storage
            save_figs (bool): Whether to save intermediate plots (currently unused)
            
        Returns:
            tuple: Best insertion, center, and extraction points
        """
        # Calculate initial suture count estimate
        num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1))
        num_sutures_initial = int(num_sutures_initial / 4)  # Adjusted for suture width calculations
        print(f"NUM SUTURES INITIAL: {num_sutures_initial}")
        
        # Set up suture range for testing
        start_range = max(2, int(num_sutures_initial))
        end_range = int(2.2 * num_sutures_initial)

        # Initialize optimization frame
        _optFrame.set_suture_range(start_range, end_range)
        _optFrame.set_distance_calculator(self.DistanceCalculator)
        
        # Storage for results
        d = {}  # Loss data dictionary
        points_dict = {}  # Optimized point positions
        self.progress_incre = (1 / (end_range - start_range + 1)) / 10
        _optFrame.start_range = start_range
        _optFrame.end_range = end_range
        _optFrame.final_suture_colors = {}
        
        # Test each suture count in range
        for num_sutures in range(start_range, end_range):
            print(f'TESTING NUM SUTURES: {num_sutures}')
            
            # Update progress GUI
            _optFrame.update_cur_sutures(num_sutures)
            self.progress = (num_sutures - start_range) / (end_range - start_range + 1)
            
            d[num_sutures] = {}
            wound_points = np.linspace(0, 1, num_sutures)
            
            # Optimize suture placement
            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = \
                self.optimize(wound_points=wound_points, optFrame=_optFrame)
            
            # Mark prioritized sutures (near high-curvature points)
            final_suture_colors = ['black'] * num_sutures
            high_curv_sutures = [0] * num_sutures
            closest_sutures = self.Constraints.get_closest_suture(ts)
            for i in closest_sutures:
                final_suture_colors[i] = '#01fd00'  # Green for prioritized
                high_curv_sutures[i] = 1
            
            _optFrame.final_suture_colors[num_sutures] = final_suture_colors
            _optFrame.high_curv_sutures = high_curv_sutures
            _optFrame.num_high_curv_sutures = np.sum(high_curv_sutures)

            # Store suture plans
            _optFrame.planned_insert_pts.append(insert_pts)
            _optFrame.planned_center_pts.append(center_pts)
            _optFrame.planned_extract_pts.append(extract_pts)

            # Calculate adaptive length sutures (extended for open wound visualization)
            dists_to_wound = []
            for cpt in center_pts:
                min_dist = 10000
                scale_len = 0
                # Find closest centerline point and get distance to wound edge
                for opt in _optFrame.parent_root.ordered_pts_dist:
                    temp_dist = math.sqrt((opt[0] - cpt[1])**2 + (opt[1] - cpt[0])**2)
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                        scale_len = opt[2]
                dists_to_wound.append(scale_len)

            def extend_sutures(insert_pts, extract_pts, center_pts, dists_to_wound):
                """Extend sutures outward for open wound visualization."""
                n_insert_pts = []
                n_extract_pts = []
                for i in range(len(center_pts)):
                    vect = (extract_pts[i][0] - insert_pts[i][0], 
                           extract_pts[i][1] - insert_pts[i][1])
                    scale_factor = dists_to_wound[i] * 0.1
                    if scale_factor < 1.0:
                        scale_factor = 1.0
                    # Extend insertion point outward
                    new_pt = (insert_pts[i][0] + (scale_factor * vect[0]), 
                             insert_pts[i][1] + (scale_factor * vect[1]))
                    n_insert_pts.append(new_pt)
                    # Extend extraction point outward
                    new_pt = (extract_pts[i][0] - (scale_factor * vect[0]), 
                             extract_pts[i][1] - (scale_factor * vect[1]))
                    n_extract_pts.append(new_pt)
                
                return n_extract_pts, n_insert_pts
            
            n_insert_pts, n_extract_pts = extend_sutures(
                insert_pts, extract_pts, center_pts, dists_to_wound
            )
            _optFrame.planned_n_insert_pts.append(n_insert_pts)
            _optFrame.planned_n_extract_pts.append(n_extract_pts)

            # Update progress
            self.progress += self.progress_incre
            _optFrame.update_progress(self.progress)

            # Calculate loss
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists
            best_loss = self.RewardFunction.hyperLoss()

            self.progress += self.progress_incre
            _optFrame.update_progress(self.progress)
            
            # Get individual loss components
            closure_loss = self.RewardFunction.lossClosureForce(1, 0)
            shear_loss = self.RewardFunction.lossClosureForce(0, 1)

            self.progress += self.progress_incre
            _optFrame.update_progress(self.progress)

            center_var_loss = self.RewardFunction.lossVar(1, 0)
            ins_ext_var_loss = self.RewardFunction.lossVar(0, 1)
            ideal_loss = self.RewardFunction.lossIdeal()

            self.progress += self.progress_incre
            _optFrame.update_progress(self.progress)
            
            print(f'loss: {best_loss}')
            print(f'closure loss: {closure_loss}')
            print(f'shear loss: {shear_loss}')

            # Store losses for plotting
            _optFrame.total_array.append(best_loss)
            _optFrame.closure_array.append(closure_loss)
            _optFrame.shear_array.append(shear_loss)
            
            # Update GUI
            _optFrame.update_losses(best_loss, closure_loss, shear_loss)
            _optFrame.update_visualization(ts, f'Suture Plan: {num_sutures} Sutures')

            # Store results
            d[num_sutures]['loss'] = best_loss
            d[num_sutures]['closure loss'] = closure_loss
            d[num_sutures]['shear loss'] = shear_loss
            d[num_sutures]['var loss - center'] = center_var_loss
            d[num_sutures]['var loss - ins/ext'] = ins_ext_var_loss
            d[num_sutures]['ideal loss'] = ideal_loss
            
            b_insert_pts, b_center_pts, b_extract_pts, b_ts = \
                insert_pts, center_pts, extract_pts, ts
            
            self.insert_pts = b_insert_pts
            self.center_pts = b_center_pts
            self.extract_pts = b_extract_pts

            # Track best result
            if best_loss < self.b_loss:
                self.b_loss = best_loss
                self.b_insert_pts = b_insert_pts
                self.b_center_pts = b_center_pts
                self.b_extract_pts = b_extract_pts

            points_dict[num_sutures] = b_ts

        # Mark optimization complete
        _optFrame.mark_complete()
        
        # Save results
        dict_to_csv(d, "clicked_losses")
        save_dict_to_file(points_dict, "clicked_points.txt")
        
        return b_insert_pts, b_center_pts, b_extract_pts


def save_dict_to_file(dic, filename):
    """
    Save a dictionary to a text file.
    
    Args:
        dic (dict): Dictionary to save
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write(str(dic))


def load_dict_from_file():
    """
    Load a dictionary from a text file.
    
    Returns:
        dict: Loaded dictionary
    """
    with open('dict.txt', 'r') as f:
        data = f.read()
    return eval(data)


def dict_to_csv(d, filename):
    """
    Convert a dictionary of loss data to CSV format.
    
    Args:
        d (dict): Dictionary with suture counts as keys and loss data as values
        filename (str): Output CSV filename (without .csv extension)
    """
    rows = []
    for k, v in d.items():
        row = {"num_sutures": k}
        row.update(v)
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=['num_sutures', 'loss', 'closure loss', 'shear loss', 'var loss'])
    df = df.sort_values(by=['loss'])
    df.to_csv(filename + ".csv", index=False)
