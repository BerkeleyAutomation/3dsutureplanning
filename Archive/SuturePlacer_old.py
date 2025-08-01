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
from progress_gui import get_progress_gui
import threading
import time


class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
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

        self.updatelabel = ''

    def optimize(self, wound_points):
        print('in optimization method')
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists

        self.Constraints.wound_points = wound_points
        
        def jac(t):
            return optim.approx_fprime(t, final_loss)

        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
            self.RewardFunction.wound_points = t
            self.RewardFunction.suture_points = list(zip(insert_pts, center_pts, extract_pts))
            return self.RewardFunction.final_loss(c_lossMin=self.c_lossMin, c_lossIdeal = self.c_lossIdeal, c_lossVarCenter = self.c_lossVarCenter, c_lossVarInsExt=self.c_lossVarInsExt, c_lossClosure = self.c_lossClosure, c_lossShear = self.c_lossShear)

        print('pre results')
        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol=1e-2, jac = jac)
        print('results calculated')
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(result.x)
        
        print('mid optimize method')

        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol = 1e-2, jac = jac)
        # plt.clf()
        # save_intermittent_plots = True
        # if save_intermittent_plots:
        #     self.DistanceCalculator.plot(result.x, "closure", plot_type='closure', save_fig='s1/' + str(len(wound_points)) + '_closure_' + str(random.randint(0, 1000000)))
        #     self.DistanceCalculator.plot(result.x, "shear", plot_type='shear', save_fig='s1/' + str(len(wound_points)) + '_shear_' + str(random.randint(0, 1000000)))

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x
    
    def test_sutures(self, num_sutures, d, losses, points_dict, save_figs=True):

        print('Testing sutures thread with: ' + str(num_sutures) + ' sutures.')
        d[num_sutures] = {}
        heuristic = num_sutures
        best_loss = float('inf')
        wound_points = np.linspace(0, 1, num_sutures)
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        best_loss = self.RewardFunction.hyperLoss()
            
        # Get individual loss components
        closure_loss = self.RewardFunction.lossClosureForce(1, 0)
        shear_loss = self.RewardFunction.lossClosureForce(0, 1)
        center_var_loss = self.RewardFunction.lossVar(1, 0)
        ins_ext_var_loss = self.RewardFunction.lossVar(0, 1)
        ideal_loss = self.RewardFunction.lossIdeal()
            
        print('loss: ', best_loss)
        print('closure loss', closure_loss)
        print('shear loss', shear_loss)
        print('center var loss', center_var_loss)
        print('InsExt var loss', ins_ext_var_loss)
        print('ideal loss', ideal_loss)
            
        # # Update progress GUI with loss information
        # if progress_gui:
        #     progress_gui.update_losses(best_loss, closure_loss, shear_loss, 
        #                              center_var_loss, ins_ext_var_loss, ideal_loss)
        #     # Update visualization with current suture plan
        #     progress_gui.update_visualization(ts, f"Optimized {num_sutures} Sutures")
            
        d[num_sutures]['loss'] = best_loss
        d[num_sutures]['closure loss'] = closure_loss
        d[num_sutures]['shear loss'] = shear_loss
        d[num_sutures]['var loss - center'] = center_var_loss
        d[num_sutures]['var loss - ins/ext'] = ins_ext_var_loss
        d[num_sutures]['ideal loss'] = ideal_loss
        b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
        losses[best_loss] = num_sutures
     
        self.insert_pts = b_insert_pts
        self.center_pts = b_center_pts
        self.extract_pts = b_extract_pts

        if best_loss < self.b_loss:
            self.b_loss = best_loss
            self.b_insert_pts = b_insert_pts
            self.b_center_pts = b_center_pts
            self.b_extract_pts = b_extract_pts

        print(losses)

        # if save_figs:
            
        #     self.DistanceCalculator.plot(b_ts, "Number of Sutures: " + str(num_sutures) + ". Total loss: " + str(best_loss), save_fig=str(num_sutures), plot_type='sutures',save_dir='clicking/'+dt_string)
        #     self.DistanceCalculator.plot(b_ts, "Closure force for " + str(num_sutures) + " sutures", save_fig= str(num_sutures), plot_type='closure', save_dir='clicking/'+dt_string)
        #     self.DistanceCalculator.plot(b_ts, "Shear force for " + str(num_sutures) + " sutures", save_fig=str(num_sutures), plot_type='shear', save_dir='clicking/'+dt_string)

        points_dict[num_sutures] = b_ts
        print('TEST SUTURES finished with ' + str(num_sutures) + '! Continuing...')

    def place_sutures(self, mainGUI, save_figs=True):

        #self.mainGUI = mainGUI
        # make a folder to store info

        # if save_figs:
        #     if not os.path.isdir("clicking"):
        #         os.mkdir('clicking')
            
        #     now = datetime.now()
        #     # dd/mm/YY H:M:S
        #     dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        #     os.mkdir('clicking/' + dt_string)
        #     os.mkdir('clicking/' + dt_string + '/sutures')
        #     os.mkdir('clicking/' + dt_string + '/closure')
        #     os.mkdir('clicking/' + dt_string + '/shear')

        # Get progress GUI if available
        #progress_gui = get_progress_gui()

        num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1)) # heuristic
        print("NUM SUTURES INITIAL", num_sutures_initial)
        
        # Set up suture range for progress tracking
        start_range = max(2, int(num_sutures_initial))
        #end_range = int(2.2 * num_sutures_initial)
        end_range = int(2 + num_sutures_initial)

        print('Optimization process will test ' + str(start_range) + ' to ' + str(end_range) + ' sutures.')
        mainGUI.after(100,mainGUI.progress_label.configure(text='Optimization process will test ' + str(start_range) + ' to ' + str(end_range) + ' sutures.'))
        #time.sleep(1)
        #mainGUI.update_idletasks()
        # if progress_gui:
        #     progress_gui.set_suture_range(start_range, end_range)
        #     progress_gui.set_status("Starting suture placement optimization...")
        #     # Set up visualization
        #     progress_gui.set_distance_calculator(self.DistanceCalculator)

        # trying threading

        # def update_gui_progress(self):
        #     self.mainGUI.progress_label.configure(text=self.updatelabel)

        self.d = {}
        self.losses = {}
        self.points_dict = {}

        for num_sutures in range(start_range, end_range): # This should be (0.8 * heuristic to 1.4 * heuristic)
            print('NUM SUTURES: ', num_sutures)
            
            #mainGUI.progress_label.configure(text="Testing " + str(num_sutures) + ' sutures...')
            #mainGUI.update_idletasks()
            #self.updatelabel = 'Testing ' + str(num_sutures) + ' sutures...'
            mainGUI.after(50, mainGUI.progress_label.configure(text='Testing ' + str(num_sutures) + ' sutures...'))
            #mainGUI.after(50, mainGUI.progress_label.configure(text=self.updatelabel))
            print('update complete')

            suture_thread = threading.Thread(target=self.test_sutures, args=(num_sutures, self.d, self.losses, self.points_dict))
            print('starting test thread')
            suture_thread.start()

            #mainGUI.update_idletasks()
            pbvalue = (num_sutures-start_range)/(end_range-start_range)
            mainGUI.progress_queue.put(pbvalue)
            print('Progresss: ' + str(pbvalue*100) + '%')
            #mainGUI.after(1,mainGUI.progress_bar.set(pbvalue))
            suture_thread.join()
            print('test thread finished')
        

        #     # # Update progress GUI
        #     # if progress_gui:
        #     #     progress_gui.update_current_sutures(num_sutures)
        #     #     progress_gui.set_status(f"Optimizing placement for {num_sutures} sutures...")
            
        #     d[num_sutures] = {}
        #     heuristic = num_sutures
        #     best_loss = float('inf')
        #     wound_points = np.linspace(0, 1, num_sutures)
        #     insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
        #     self.RewardFunction.insert_dists = insert_dists
        #     self.RewardFunction.center_dists = center_dists
        #     self.RewardFunction.extract_dists = extract_dists
        #     best_loss = self.RewardFunction.hyperLoss()
            
        #     # Get individual loss components
        #     closure_loss = self.RewardFunction.lossClosureForce(1, 0)
        #     shear_loss = self.RewardFunction.lossClosureForce(0, 1)
        #     center_var_loss = self.RewardFunction.lossVar(1, 0)
        #     ins_ext_var_loss = self.RewardFunction.lossVar(0, 1)
        #     ideal_loss = self.RewardFunction.lossIdeal()
            
        #     print('loss: ', best_loss)
        #     print('closure loss', closure_loss)
        #     print('shear loss', shear_loss)
        #     print('center var loss', center_var_loss)
        #     print('InsExt var loss', ins_ext_var_loss)
        #     print('ideal loss', ideal_loss)
            
        #     # # Update progress GUI with loss information
        #     # if progress_gui:
        #     #     progress_gui.update_losses(best_loss, closure_loss, shear_loss, 
        #     #                              center_var_loss, ins_ext_var_loss, ideal_loss)
        #     #     # Update visualization with current suture plan
        #     #     progress_gui.update_visualization(ts, f"Optimized {num_sutures} Sutures")
            
        #     d[num_sutures]['loss'] = best_loss
        #     d[num_sutures]['closure loss'] = closure_loss
        #     d[num_sutures]['shear loss'] = shear_loss
        #     d[num_sutures]['var loss - center'] = center_var_loss
        #     d[num_sutures]['var loss - ins/ext'] = ins_ext_var_loss
        #     d[num_sutures]['ideal loss'] = ideal_loss
        #     b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
        #     losses[best_loss] = num_sutures
     
        #     self.insert_pts = b_insert_pts
        #     self.center_pts = b_center_pts
        #     self.extract_pts = b_extract_pts

        #     if best_loss < self.b_loss:
        #         self.b_loss = best_loss
        #         self.b_insert_pts = b_insert_pts
        #         self.b_center_pts = b_center_pts
        #         self.b_extract_pts = b_extract_pts

        #     print(losses)

        #     if save_figs:
            
        #         self.DistanceCalculator.plot(b_ts, "Number of Sutures: " + str(num_sutures) + ". Total loss: " + str(best_loss), save_fig=str(num_sutures), plot_type='sutures',save_dir='clicking/'+dt_string)
        #         self.DistanceCalculator.plot(b_ts, "Closure force for " + str(num_sutures) + " sutures", save_fig= str(num_sutures), plot_type='closure', save_dir='clicking/'+dt_string)
        #         self.DistanceCalculator.plot(b_ts, "Shear force for " + str(num_sutures) + " sutures", save_fig=str(num_sutures), plot_type='shear', save_dir='clicking/'+dt_string)

        #     points_dict[num_sutures] = b_ts

        # # Mark optimization as complete
        # # if progress_gui:
        # #     progress_gui.mark_complete()
        # #     progress_gui.set_status("Optimization complete! Saving results...")
        
        mainGUI.progress_queue.put("DONE")
        print('Optimization Complete.')
        mainGUI.progress_bar.set(1)
        dict_to_csv(self.d, "clicked_losses")
        save_dict_to_file(self.points_dict, "clicked_points.txt")
        return self.insert_pts, self.center_pts, self.extract_pts

def save_dict_to_file(dic, filename):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
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