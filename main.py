# Import Packages
import customtkinter as ctk
from tkinter import filedialog
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import webbrowser

# from InsertionPointGenerator import InsertionPointGenerator
# from ScaleGenerator import ScaleGenerator
# from SutureDisplayAdjust2D import SutureDisplayAdjust2D
# import RewardFunction
import numpy as np
import cv2
import math
from PIL import Image, ImageTk, ImageDraw
import json

import tkinter as tk

import EdgeDetector
from PIL import Image

# import threading
# import time
# import queue
import sys

class OptFrame(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.grid_columnconfigure(0, weight=1)

        #initial widgets to frame
        self.cur_num_sutures = 0
        self.test_suture_range = (0,0)
        self.best_loss = float('inf')
        self.normalized_best_loss = float('inf')
        self.best_num_sutures = 0

        # storing calculated suture plans
        self.planned_insert_pts = []
        self.planned_center_pts = []
        self.planned_extract_pts = []
        self.planned_n_insert_pts = []
        self.planned_n_extract_pts = []
        self.start_range = 0
        self.end_range = 0

        self.total_array = []
        self.closure_array = []
        self.shear_array = []
        
        self.cur_visualization_data = None
        self.distance_calc = None
        self.parent_root = parent

        # LEFT SIDE
        # frame for optimization details
        self.infopanel = ctk.CTkFrame(self, fg_color="#7dafce")
        self.infopanel.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.infopanel.grid_rowconfigure(0,weight=1)
        self.infopanel.grid_columnconfigure(0,weight=1)
        self.infopanel.grid_columnconfigure(2,weight=1)

        infoheader = ctk.CTkLabel(self.infopanel, text='Optimization Progress Details', font=('Arial',24,'bold'),justify='center',text_color='#003049')
        infoheader.grid(row=0,column=1,padx=10,pady=10,sticky='ew')
        progress_label = ctk.CTkLabel(self.infopanel, text='Current Progress', font=('Arial',24),justify='center',text_color='#003049')
        progress_label.grid(row=1,column=1,padx=10,pady=10,sticky='ew')

        # left panel progress bar
        self.progress_bar = ctk.CTkProgressBar(self.infopanel, orientation='horizontal', mode='determinate', width=300, progress_color='#780000', fg_color='#f8f9fa')
        self.progress_bar.grid(row=2,column=1,padx=10,pady=10)
        self.progress_bar.set(0)
        self.progress_percent = ctk.CTkLabel(self.infopanel,text='0%',font=('Arial',24,'bold'),justify='center', text_color='#003049')
        self.progress_percent.grid(row=3,column=1,padx=5,pady=5)

        self.cur_suture_info = ctk.CTkLabel(self.infopanel, text='Calculating loss for placement plan with 0 sutures...', font=('Arial',17),justify='center', text_color='#003049')
        self.cur_suture_info.grid(row=4,column=1,padx=5,pady=5)

        # loss information
        self.loss_frame = ctk.CTkFrame(self.infopanel, fg_color="#7dafce")
        self.loss_frame.grid(row=5,column=1,padx=10,pady=10,sticky='ew')

        self.loss_frame.grid_rowconfigure(0,weight=1)
        self.loss_frame.grid_columnconfigure(0,weight=1)
        self.loss_frame.grid_columnconfigure(2,weight=1)

        self.total_loss_label = ctk.CTkLabel(self.loss_frame, text='Total Loss: --', font=('Arial', 17),justify='center', text_color='#003049')
        #self.total_loss_label.grid(row=0,column=1,padx=10,sticky='ew')
        self.closure_loss_label = ctk.CTkLabel(self.loss_frame, text='Closure Loss: --', font=('Arial',17),justify='center', text_color='#003049')
        #self.closure_loss_label.grid(row=1,column=1,padx=10,sticky='ew')
        self.shear_loss_label = ctk.CTkLabel(self.loss_frame, text='Shear Loss: --', font=('Arial',17),justify='center', text_color='#003049')
        #self.shear_loss_label.grid(row=2,column=1,padx=10,sticky='ew')

        # loss plots
        self.loss_plot_frame = ctk.CTkFrame(self.loss_frame,fg_color="#7dafce")
        self.loss_plot_frame.grid(row=3, column=1,padx=10,sticky='ew')

        self.loss_fig = Figure(figsize=(6,2), dpi=80)
        self.total_ax = self.loss_fig.add_subplot(111)

        self.loss_fig_canvas = FigureCanvasTkAgg(self.loss_fig, self.loss_plot_frame)
        self.loss_fig_canvas.get_tk_widget().grid(row=0,column=0,padx=0,pady=0)

        self.total_ax.text(0.5,0.5,'Waiting for loss...',ha='center',va='center',transform=self.total_ax.transAxes,fontsize=12)
        self.total_ax.axis('off')
        self.loss_fig_canvas.draw()
        
        # best result information
        self.best_frame = ctk.CTkFrame(self.infopanel,fg_color='#f8f9fa') #'#2fc986'
        self.best_frame.grid(row=6,column=1,padx=10,pady=10,sticky='ew')

        self.best_frame.grid_rowconfigure(0,weight=1)
        self.best_frame.grid_columnconfigure(0,weight=1)
        self.best_frame.grid_columnconfigure(2,weight=1)

        self.best_label = ctk.CTkLabel(self.best_frame, text='Best Result: --', font=('Arial',17,'bold'),justify='center',fg_color='#f8f9fa', text_color='#003049')
        self.best_label.grid(row=0,column=1,padx=10,pady=10,sticky='ew')

        # RIGHT SIDE
        # frame for optimization graph visualization
        self.graphpanel = ctk.CTkFrame(self, fg_color="#7dafce")
        self.graphpanel.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.graph_title = ctk.CTkLabel(self.graphpanel, text='Current Suture Plan Visualization', font=('Arial',24,'bold'), text_color='#003049')
        self.graph_title.grid(row=0,column=0,padx=10,pady=10)

        self.graph_fig = Figure(figsize=(4,4), dpi=80)
        self.graph_ax = self.graph_fig.add_subplot(111)
        
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, self.graphpanel)
        self.graph_canvas.get_tk_widget().grid(row=1,column=0,padx=10,pady=10)

        # inital plot
        self.graph_ax.text(0.5,0.5,'Waiting for optimization...',ha='center',va='center',transform=self.graph_ax.transAxes,fontsize=17)
        self.graph_ax.set_xlim(0,1)
        self.graph_ax.set_ylim(0,1)
        self.graph_ax.axis('off')
        self.graph_canvas.draw()

    def update_progress(self,progress: float, stage: str=''):
        self.progress_bar.set(progress)
        self.progress_percent.configure(text=f'{progress*100:.1f}%')
        self.parent_root.update()
    
    def set_suture_range(self,start_range: int, end_range: int):
        self.test_suture_range = (start_range, end_range)

    def update_cur_sutures(self, num_sutures: int):
        # update current number of sutures being optimized
        self.cur_num_sutures = num_sutures

        #self.cur_suture_info.configure(text=f'Testing: {num_sutures} sutures')
        self.cur_suture_info.configure(text=f'Calculating loss for placement plan with {num_sutures} sutures...')
        if self.test_suture_range[1] > 0:
            progress = (num_sutures - self.test_suture_range[0]) / (self.test_suture_range[1] - self.test_suture_range[0] + 1)
            self.update_progress(progress)
    
    def update_losses(self, total_loss: float, closure_loss: float, shear_loss: float):
        # update max and min losses in parent GUI
        if total_loss > self.parent_root.max_total_loss:
            self.parent_root.max_total_loss = total_loss
            print('max_total_loss: ' + str(self.parent_root.max_total_loss))
        if total_loss < self.parent_root.min_total_loss:
            self.parent_root.min_total_loss = total_loss
            print('min_total_loss: ' + str(self.parent_root.min_total_loss))
        if closure_loss > self.parent_root.max_closure_loss:
            self.parent_root.max_closure_loss = closure_loss
            print('max_closure_loss: ' + str(self.parent_root.max_closure_loss))
        if closure_loss < self.parent_root.min_closure_loss:
            self.parent_root.min_closure_loss = closure_loss
            print('min_closure_loss: ' + str(self.parent_root.min_closure_loss))
        if shear_loss > self.parent_root.max_shear_loss:
            self.parent_root.max_shear_loss = shear_loss
            print('max_shear_loss: ' + str(self.parent_root.max_shear_loss))
        if shear_loss < self.parent_root.min_shear_loss:
            self.parent_root.min_shear_loss = shear_loss
            print('min_shear_loss: ' + str(self.parent_root.min_shear_loss))
        
        # calculate normalized loss percentages
        normalized_total_loss = ((total_loss - self.parent_root.min_total_loss) / (self.parent_root.max_total_loss - self.parent_root.min_total_loss)) * 100
        normalized_closure_loss = ((closure_loss - self.parent_root.min_closure_loss) / (self.parent_root.max_closure_loss - self.parent_root.min_closure_loss)) * 100
        normalized_shear_loss = ((shear_loss - self.parent_root.min_shear_loss) / (self.parent_root.max_shear_loss - self.parent_root.min_shear_loss)) * 100

        # update display of loss information
        # self.total_loss_label.configure(text=f'Total Loss: {total_loss:.0f}')
        # self.closure_loss_label.configure(text=f'Closure Loss: {closure_loss:.0f}')
        # self.shear_loss_label.configure(text=f'Shear Loss: {shear_loss:.0f}')
        self.total_loss_label.configure(text=f'Total Loss: {normalized_total_loss:.2f}%')
        self.closure_loss_label.configure(text=f'Closure Loss: {normalized_closure_loss:.2f}%')
        self.shear_loss_label.configure(text=f'Shear Loss: {normalized_shear_loss:.2f}%')

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.normalized_best_loss = normalized_total_loss
            self.best_sutures = self.cur_num_sutures
            self.best_label.configure(text=f'Best Result: {self.best_sutures} sutures - Loss: {self.normalized_best_loss:.2f}%')
        
        # plotting losses over tested num sutures
        self.total_ax.clear()
        loss_X = np.array(list(range(self.start_range, self.start_range+len(self.total_array))))

        self.total_ax.plot(loss_X, np.array(self.total_array)/max(self.total_array), color='red',linewidth=1.5)
        self.total_ax.plot(loss_X, np.array(self.closure_array)/max(self.closure_array), color='blue',linewidth=1.5)
        self.total_ax.plot(loss_X, np.array(self.shear_array)/max(self.shear_array), color='black',linewidth=1.5)
        self.total_ax.set_title('Total, Closure, and Shear Loss', fontsize=12)
        self.total_ax.legend(["Total", "Closure", "Shear"], loc="lower left")
        self.total_ax.grid(True,alpha=0.2)
        self.total_ax.set_xlabel('# Sutures')
        self.total_ax.set_ylabel('Loss')
        self.total_ax.set_xticks(loss_X)

        # # update canvas
        self.loss_fig_canvas.draw()
        self.parent_root.update()
    
    def set_distance_calculator(self,distance_calculator):
        self.distance_calc = distance_calculator

    def update_visualization(self, wound_point_t, title='Suture Plan'):
        # clear current plot
        self.graph_ax.clear()
        
        # get wound curve points for plotting
        num_pts = len(wound_point_t)
        self.num_pts = num_pts

        # get curve and gradient for each point
        wound_points, wound_curve = self.distance_calc.wound_parametric(wound_point_t,0)
        wound_derivatives_x, wound_derivatives_y = self.distance_calc.wound_parametric(wound_point_t, 1)

        # calculate insertion and extraction points
        def get_norm(x,y):
            return math.sqrt(x**2 + y**2)
        
        norms = [get_norm(wound_derivatives_x[i],wound_derivatives_y[i]) for i in range(len(wound_derivatives_x))]
        normal_vecs = [[wound_derivatives_y[i]/norms[i],-wound_derivatives_x[i]/norms[i]] for i in range(num_pts)]
        normal_vecs = [[normal_vec[0] * self.distance_calc.wound_width,normal_vec[1] * self.distance_calc.wound_width] for normal_vec in normal_vecs]

        insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]
        extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in range(num_pts)]
        center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]

        # adaptive length sutures
        dists_to_wound = []
        for cpt in center_pts:
            min_dist = 10000
            scale_len = 0
            #find point in centerline that is closest to center point, save distance to outside of wound
            for opt in self.parent_root.ordered_pts_dist:
                temp_dist = math.sqrt((opt[0]-cpt[1])**2 + (opt[1]-cpt[0])**2)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    scale_len = opt[2]
            dists_to_wound.append(scale_len)

        def extend_sutures(insert_pts, extract_pts, center_pts, dists_to_wound):
            n_insert_pts = []
            n_extract_pts = []
            for i in range(len(center_pts)):
                vect = (extract_pts[i][0]-insert_pts[i][0], extract_pts[i][1]-insert_pts[i][1])
                scale_factor = dists_to_wound[i]*0.1
                if scale_factor < 1.0:
                    scale_factor = 1.0
                new_pt = (insert_pts[i][0] + ((scale_factor) * vect[0]), insert_pts[i][1] + ((scale_factor) * vect[1]))
                n_insert_pts.append(new_pt)
                new_pt = (extract_pts[i][0] - ((scale_factor) * vect[0]), extract_pts[i][1] - ((scale_factor) * vect[1]))
                n_extract_pts.append(new_pt)
            
            return n_extract_pts, n_insert_pts

        # plot the wound curve
        X_, Y_ = [], []
        for i in range(500):
            t = min(wound_point_t) + (max(wound_point_t) - min(wound_point_t))*i/500
            temp = self.distance_calc.wound_parametric(t,0)
            X_.append(temp[1])
            Y_.append(-temp[0])
        self.graph_ax.plot(X_,Y_, color='black',linewidth=1.5,alpha=0.7)

        self.insert_pts = insert_pts
        self.extract_pts = extract_pts
        self.center_pts = center_pts

        n_insert_pts, n_extract_pts = extend_sutures(insert_pts, extract_pts, center_pts, dists_to_wound)
        self.n_insert_pts = n_insert_pts
        self.n_extract_pts = n_extract_pts

        # # plot suture points
        # self.graph_ax.scatter([n_insert_pts[i][1] for i in range(num_pts)],[-n_insert_pts[i][0] for i in range(num_pts)],c='red',s=30,alpha=0.8,label='Insertion')
        # self.graph_ax.scatter([n_extract_pts[i][1] for i in range(num_pts)],[-n_extract_pts[i][0] for i in range(num_pts)],c='blue',s=30,alpha=0.8,label='Extraction')
        # # draw suture lines on plot
        # for i in range(len(n_insert_pts)):
        #     self.graph_ax.plot([n_insert_pts[i][1],n_extract_pts[i][1]],[-n_insert_pts[i][0],-n_extract_pts[i][0]],color='black',linewidth=1,alpha=0.6)

        # plot suture points
        self.graph_ax.scatter([insert_pts[i][1] for i in range(num_pts)],[-insert_pts[i][0] for i in range(num_pts)],c='red',s=30,alpha=0.8,label='Insertion')
        self.graph_ax.scatter([extract_pts[i][1] for i in range(num_pts)],[-extract_pts[i][0] for i in range(num_pts)],c='blue',s=30,alpha=0.8,label='Extraction')
        #self.graph_ax.scatter([center_pts[i][1] for i in range(num_pts)],[-center_pts[i][0] for i in range(num_pts)],c='green',s=30,alpha=0.8,label='Center')

        # draw suture lines on plot
        for i in range(len(insert_pts)):
            self.graph_ax.plot([insert_pts[i][1],extract_pts[i][1]],[-insert_pts[i][0],-extract_pts[i][0]],color='black',linewidth=1,alpha=0.6)
        
        #self.graph_ax.set_title(f'{title}\n{self.cur_num_sutures} sutures - Loss: {self.best_loss:.0f}', fontsize=17)
        self.graph_ax.set_title(f'{title}\nLoss: {self.normalized_best_loss:.2f}%', fontsize=13)
        self.graph_ax.axis('square')
        self.graph_ax.grid(True,alpha=0.2)
        self.graph_ax.legend(loc='upper right',fontsize=8)

        # remove axis labels
        self.graph_ax.set_xlabel('')
        self.graph_ax.set_ylabel('')
        self.graph_ax.tick_params(axis='x',labelbottom=False)
        self.graph_ax.tick_params(axis='y',labelleft=False)

        # update canvas
        self.graph_canvas.draw()
        self.parent_root.update()
    
    def mark_complete(self):
        self.update_progress(1.0)
        self.parent_root.update()
        print('Optimization Complete!')
        print(f'Best results: {self.best_sutures} sutures with {self.normalized_best_loss:.2f}% loss')

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('System')
        #ctk.set_default_color_theme('green')

        self.title('Suture-It')
        self.geometry('1300x840') #'750x750'
        self.grid_columnconfigure(0, weight=1)

        link_icon = Image.open("external_link.png")
        self.link_image = ctk.CTkImage(link_icon, size=(15,15))
        copyr = ctk.CTkButton(self, text='This is a Beta Test of software in-progress. Please send feedback to: cassie.jeng@berkeley.edu', image=self.link_image, compound='right', font=('Arial',15), command=self.open_email, fg_color='#f8f9fa', text_color='#003049', hover_color="#dee2e6")
        #copyr = ctk.CTkButton(self, text='Interface Beta Test -- Please share only with permission from AUTOLab (cassie.jeng@berkeley.edu)', image=self.link_image, compound='right', font=('Arial',15), command=self.open_email, fg_color='#f8f9fa', text_color='#003049', hover_color="#dee2e6")
        copyr.grid(row=0, column=0, padx=0, pady=0, sticky='ew')

        suture_planner_title = ctk.CTkLabel(self, text='Suture-It', font=('Arial Bold',50), fg_color='#7dafce', text_color='#003049')
        suture_planner_title.grid(row=1, column=0, padx=20, pady=10, sticky='ew')

        # autolab logo button
        autolab_image = Image.open("logo2.png")
        self.company_image = ctk.CTkImage(autolab_image, size=(180,45)) #148,45
        self.suture_it_company = ctk.CTkButton(self, text='', image=self.company_image, compound='left', command=self.open_link, fg_color='#f8f9fa', text_color='#003049', hover_color="#dee2e6")
        self.suture_it_company.grid(row=2, column=0, padx=5, pady=5)

        # Welcome to Suture-It. Upload a wound image to start!
        self.suture_planner_text = ctk.CTkLabel(self, text='Welcome to Suture-It!', font=('Arial',24), wraplength=700, text_color='#003049')
        self.suture_planner_text.grid(row=3, column=0, padx=20, pady=10)

        self.disclaimer = ctk.CTkLabel(self, text='Disclaimer:\nIn its current stage, Suture-It does not consider factors such as:\n   - Potential skin deformations during implementation\n   - Dynamic skin surface and patient movement during procedure\n   - Differences in equipment materials (needle, thread, etc.)\n   - Depth/3D conceptualization of wound\n\nWe recognize future work that could complement the current program:\n   - Dynamic adjustments to suture plan during suturing\n   - Projection of suture plan onto wound for real-time guidance', compound='left',justify='left', anchor='w',font=('Arial',17), wraplength=700, text_color='#003049')
        self.disclaimer.grid(row=4, column=0, padx=20, pady=10)

        results_image = Image.open("resulting_sutures.png")
        self.results_img = ctk.CTkImage(results_image, size=(835,300)) #2444 × 878
        self.results_fig = ctk.CTkLabel(self, text="", image=self.results_img)
        self.results_fig.grid(row=5, column=0, padx=20, pady=0)

        example_images = Image.open("example_uploads.png")
        self.example_img = ctk.CTkImage(example_images, size=(445,300))
        self.example_fig = ctk.CTkLabel(self, text="", image=self.example_img)

        self.results_cap = ctk.CTkLabel(self, text='Example Suture-It Results',font=('Arial',17), wraplength=700, text_color='#003049')
        #self.results_cap.grid(row=6, column=0, padx=20, pady=10)

        self.get_started_button = ctk.CTkButton(self, text='Start Suture-It!', command=self.progress_to_upload, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.get_started_button.grid(row=100, column=0, padx=20, pady=20)

        self.upload_inst = ctk.CTkLabel(self, text='Example wound images that can be uploaded to this program are shown below and included in the program repository. Accepted file types: *.jpg, *.jpeg, *.png, *.bmp, *.tiff, *.gif\n\nTo use custom wound images:\n   - Search \"open wound images\" on Google Images [Viewer discretion advised for results]\n   - Save chosen image locally\n   - Click \"Upload Wound Image\" below\n   - Navigate to where the image was saved and select to upload', compound='left',justify='left', anchor='w',font=('Arial',17), wraplength=700, text_color='#003049')

        self.upload_image_button = ctk.CTkButton(self, text='Upload Wound Image', command=self.upload_image, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        #self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)

        self.plan_num = 0
        self.scale_pts = [(331,120),(342,160)]
        # self.a = 0.5
        self.a = 0.77
        with open('loss.json', 'r') as lossfile:
            lossjson = json.load(lossfile)
            self.max_total_loss = lossjson['max_total_loss']
            self.min_total_loss = lossjson['min_total_loss']
            self.max_closure_loss = lossjson['max_closure_loss']
            self.min_closure_loss = lossjson['min_closure_loss']
            self.max_shear_loss = lossjson['max_shear_loss']
            self.min_shear_loss = lossjson['min_shear_loss']

        self.suture_drawn_button = ctk.CTkButton(self, text='Done!', command=self.suture_drawn, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.mask_generated = ctk.CTkButton(self, text='Compute Wound Centerline', command=self.compute_centerline, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.start_opt = ctk.CTkButton(self, text='Start Suture Planning', command=self.optimization, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.see_final_opt = ctk.CTkButton(self, text='View Optimized Suture Plan', command=self.view_final, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        
        self.buttons_frame = ctk.CTkFrame(self, fg_color='#f8f9fa')
        self.buttons_frame.grid_rowconfigure(0,weight=1)

        self.slider_frame = ctk.CTkFrame(self.buttons_frame, fg_color='#f8f9fa')
        self.slider_frame.grid_rowconfigure(0,weight=1)

        self.final_canvas_frame = ctk.CTkFrame(self, fg_color='#f8f9fa')
        self.final_canvas_frame.grid_rowconfigure(0,weight=1)

        #self.a_slider = ctk.CTkSlider(self.buttons_frame,from_=0, to=1,orientation='horizontal',width=150)
        #self.a_value = ctk.CTkLabel(self.buttons_frame, text=f'Elliptical Minor Axis = {round(self.a,2)}\nIncreasing will relax distance between sutures.',font=('Arial',12), text_color='#003049')
        
        self.rerun_button = ctk.CTkButton(self.buttons_frame, text='Rerun Suture-It', command=self.rerun, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.restart_button = ctk.CTkButton(self.buttons_frame, text='Restart Suture-It', command=self.restart, width=150, height=50, font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')
        self.end_program_button = ctk.CTkButton(self.buttons_frame, text='Close Program', command=self.on_close, width=150, height=50,font=('Arial',24),fg_color='#7dafce', text_color='#003049', hover_color='#7ea3ba')

    def on_close(self):
        #self.destroy()

        # write max and min losses to loss.json file
        lossjson = {
            "max_total_loss": self.max_total_loss,
            "min_total_loss": self.min_total_loss,
            "max_closure_loss": self.max_closure_loss,
            "min_closure_loss": self.min_closure_loss,
            "max_shear_loss": self.max_shear_loss,
            "min_shear_loss": self.min_shear_loss
        }
        with open('loss.json','w') as lossfile:
            json.dump(lossjson, lossfile, indent=4)

        self.quit()
        sys.exit()
    
    def open_link(self):
        auto_lab_url = 'https://autolab.berkeley.edu/'
        webbrowser.open_new_tab(auto_lab_url)

    def open_email(self):
        email_address = 'cassie.jeng@berkeley.edu'
        subject = 'Suture-It Interface Beta Testing'
        mail_url = f"mailto:{email_address}?subject={subject}"
        webbrowser.open_new(mail_url)
    
    def slider_update(self,event):
        self.a = self.a_slider.get()
        self.a_value.configure(text=f'Elliptical Minor Axis = {round(self.a,2)}\nIncreasing will relax distance between sutures.')

    def on_image_click(self,event):
        x, y = event.x, event.y

        # allow user to redo click (remove previously drawn suture if more than 2 pts)
        if len(self.scale_pts) >=2:
            self.scale_pts = []
            self.image_canvas.delete('suture')
        
        self.scale_pts.append((x,y))
        r = 4 #radius of circle for point
        self.image_canvas.create_oval(x-r, y-r, x+r, y+r, fill='red', outline='', tags='suture')

        if len(self.scale_pts) == 2:
            self.image_canvas.create_line(self.scale_pts[0][0], self.scale_pts[0][1], self.scale_pts[1][0], self.scale_pts[1][1], fill='black', width=2, tags='suture')
            print('point 1: (' + str(self.scale_pts[0][0]) + ',' + str(self.scale_pts[0][1]) + ')')
            print('point 2: (' + str(self.scale_pts[1][0]) + ',' + str(self.scale_pts[1][1]) + ')')
    
    def start_draw(self, event):
        # if filled poly exists, remove it
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_canvas.create_image(0,0,anchor=tk.NW,image=self.tk_image)

        # if outline exists already, remove it
        if hasattr(self, "line_ids"):
            for line_id in self.line_ids:
                self.image_canvas.delete(line_id)
        self.points = [(event.x, event.y)]
        self.line_ids = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            self.points.append((event.x, event.y))
            if len(self.points) >= 2:
                line_id = self.image_canvas.create_line(*self.points[-2], *self.points[-1], fill='blue', width=3)
                self.line_ids.append(line_id)
    
    def end_draw(self, event):
        self.drawing = False
        self.points.append((event.x, event.y))

        if len(self.points) > 2:
            line_id = self.image_canvas.create_line(*self.points[-1], *self.points[0], fill='blue', width=3)
            self.line_ids.append(line_id)

            h, w = self.tk_image.height(), self.tk_image.width()
            self.wound_mask = np.zeros((h,w), dtype=np.uint8)

            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(self.wound_mask, [pts], 255)

            base_image = self.image.convert('RGBA')
            overlay = Image.new('RGBA', (w, h), (0,0,255,0))
            for y in range(h):
                for x in range(w):
                    if self.wound_mask[y,x] == 255:
                        # if pixel is inside wound region, place semi-transparents blue pixel overtop
                        overlay.putpixel((x,y), (0,0,255,100))
                    
            blended = Image.alpha_composite(base_image,overlay)
            self.tk_image = ImageTk.PhotoImage(blended)
            self.image_canvas.create_image(0,0,anchor=tk.NW, image=self.tk_image)
            self.image_canvas.image = self.tk_image # keep a reference to image

    def suture_drawn(self):
        # remove drawn suture and ability to draw more sutures
        # self.suture_drawn_button.grid_forget()
        # self.image_canvas.unbind('<Button-1>')
        # self.image_canvas.delete('suture')

        self.suture_planner_text.configure(text='Click and drag to draw the outline of the wound! Release when done. The region should be fully enclosed. Click and drag again to redraw if needed.')

        self.drawing = False
        self.points = []
        self.line_ids = []

        self.image_canvas.bind('<ButtonPress-1>', self.start_draw)
        self.image_canvas.bind('<B1-Motion>', self.draw)
        self.image_canvas.bind('<ButtonRelease-1>', self.end_draw)

        self.mask_generated.grid(row=100, column=0, padx=20, pady=20)

    def progress_to_upload(self):
        self.disclaimer.grid_forget()
        self.results_fig.grid_forget()
        self.get_started_button.grid_forget()
        self.suture_planner_text.configure(text='Upload a wound image to start!')

        self.upload_inst.grid(row=4, column=0, padx=20, pady=10)
        self.example_fig.grid(row=5, column=0, padx=20, pady=0)
        self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)

    def upload_image(self):
        image_path = filedialog.askopenfilename(title='Select an Image', filetypes=[('Image Files','*.jpg *.jpeg *.png *.bmp *.tiff *.gif')])
        #self.disclaimer.grid_forget()
        #self.results_fig.grid_forget()
        self.upload_inst.grid_forget()
        self.example_fig.grid_forget()
        #self.results_cap.grid_forget()
        
        if image_path:
            self.image_path = image_path
            self.upload_image_button.grid_forget()
            self.image = Image.open(self.image_path).resize((600,400))
            self.tk_image = ImageTk.PhotoImage(self.image)

            self.image_canvas = ctk.CTkCanvas(self, width=600, height=400, bg='#f8f9fa', highlightthickness=0)
            self.image_canvas.grid(row=4, column=0, padx=20, pady=20)
            self.image_canvas.create_image(0,0,anchor=tk.NW,image=self.tk_image)

            # self.suture_planner_text.configure(text='Draw an example suture on the image by clicking two endpoints. The example suture should have your desired estimate suture length.')
            
            # self.image_canvas.bind('<Button-1>', self.on_image_click)
            # self.suture_drawn_button.grid(row=100, column=0, padx=20, pady=20)
            self.suture_drawn()
        else:
            print('No image selected. Exiting.')
            return
    
    def extend_hernia(self, ordered_pts, max_dist_hernia):
        extend_thresh = math.ceil((max_dist_hernia * 0.25) / 4)
        
        # extend end
        vect = (ordered_pts[-1][0] - ordered_pts[-10][0], ordered_pts[-1][1] - ordered_pts[-10][1])
        lenpts = math.sqrt((ordered_pts[-1][0] - ordered_pts[-2][0])**2 + (ordered_pts[-1][1] - ordered_pts[-2][1])**2)
        new_pts = []
        for d in np.arange(0.25, extend_thresh, 0.25):
            #new_pt = (math.ceil(ordered_pts[-1][0] + ((d / lenpts) * vect[0])), math.ceil(ordered_pts[-1][1] + ((d / lenpts) * vect[1])))
            new_pt = (ordered_pts[-1][0] + ((d / lenpts) * vect[0]), ordered_pts[-1][1] + ((d / lenpts) * vect[1]))
            new_pts.append(new_pt)
        for npts in new_pts:
            ordered_pts.append(npts)
            
        # extend beginning
        vect = (ordered_pts[0][0] - ordered_pts[10][0], ordered_pts[0][1] - ordered_pts[10][1])
        lenpts = math.sqrt((ordered_pts[0][0] - ordered_pts[1][0])**2 + (ordered_pts[0][1] - ordered_pts[1][1])**2)
        new_pts = []
        for d in np.arange(0.25, extend_thresh, 0.25):
            #new_pt = (math.ceil(ordered_pts[0][0] + ((d / lenpts) * vect[0])), math.ceil(ordered_pts[0][1] + ((d / lenpts) * vect[1])))
            new_pt = (ordered_pts[0][0] + ((d / lenpts) * vect[0]), ordered_pts[0][1] + ((d / lenpts) * vect[1]))
            new_pts.append(new_pt)
        for npts in new_pts:
            ordered_pts.insert(0,npts)
            
        print('new hernia centerline points added')
        return ordered_pts
    
    def compute_centerline(self):
        ordered_pts, _, _, max_dist_hernia, self.ordered_pts_dist = EdgeDetector.img_to_line(self.image_path, self.wound_mask)
        
        # hernia detection (wound width to centerline)
        hernia_thresh = 25.0
        if max_dist_hernia > hernia_thresh:
            print('HERNIA type wound image. max dist: ' + str(max_dist_hernia))
            # artificially extend centerline if wound is hernia type
            ordered_pts = self.extend_hernia(ordered_pts, max_dist_hernia)

        self.x = [a[1] for a in ordered_pts]
        self.y = [a[0] for a in ordered_pts]

        base_image = self.image.convert('RGB')
        draw = ImageDraw.Draw(base_image)

        # draw centerline as ellipse of points
        for pt in ordered_pts:
            draw.ellipse((pt[1]-2, pt[0]-2, pt[1]+2, pt[0]+2), fill='red')
        
        self.tk_image = ImageTk.PhotoImage(base_image)
        self.image_canvas.create_image(0,0,anchor=tk.NW,image=self.tk_image)
        self.image_canvas.image = self.tk_image # keep reference to image

        # create B-spline representation - smooth curve
        self.tck, u = inter.splprep([self.x,self.y], k=5)

        self.mask_generated.grid_forget()
        self.suture_planner_text.configure(text='Wound centerline is displayed in red. Press the button below to begin suture planning.')
        self.start_opt.grid(row=100, column=0, padx=20, pady=20)
    
    def optimization(self):
        print('Starting Optimization')
        self.start_opt.grid_forget()
        self.image_canvas.destroy()
        self.suture_planner_text.configure(text='Running Suture Placement Optimization!')

        # Start progress canvas (progress bar and update information)
        # Create instance of frame for optimization progress
        self.optFrame = OptFrame(parent=self, width=500, height=500, border_width=2, border_color='#003049', fg_color='transparent')
        self.optFrame.grid(row=50, column=0, padx=20, pady=20, sticky='nsew')

        # main optimization algorithm
        print('Starting Main Algorithm')
        wound_width = 7 # 5
        real_dist = 7 # 5
        pixel_dist = math.sqrt((self.scale_pts[0][0] - self.scale_pts[1][0])**2 + (self.scale_pts[0][1] - self.scale_pts[1][1])**2)
        mm_per_pixel = real_dist / pixel_dist
        wound_parametric = lambda t,d: inter.splev(t,self.tck,der=d)

        newSuturePlacer = SuturePlacer(wound_width,mm_per_pixel)
        newSuturePlacer.tck = self.tck
        newSuturePlacer.DistanceCalculator.tck = self.tck
        newSuturePlacer.RewardFunction.a = self.a
        # self.a_value.configure(text=f'Elliptical Minor Axis = {round(self.a,2)}\nIncreasing will relax distance between sutures.')

        newSuturePlacer.wound_parametric = wound_parametric
        newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        newSuturePlacer.image = self.image_path
        newSuturePlacer.place_sutures(_optFrame=self.optFrame)

        self.see_final_opt.grid(row=100, column=0, padx=20, pady=20)
        self.start_range = self.optFrame.start_range
        self.end_range = self.optFrame.end_range
        self.optFrame.cur_suture_info.configure(text='Complete.')
        # self.suture_planner_text.configure(text='Running Suture Placement Optimization!')
    
    def rerun(self):
        self.a = self.a_slider.get()
        print(f'Rerunning program with a = {round(self.a,2)}')
        self.final_canvas.destroy()
        self.final_canvas_ad.destroy()
        #self.final_canvas_orig.destroy()
        self.final_canvas_frame.grid_forget()
        self.plan_num_slider.grid_forget()
        self.plan_num_slider.unbind('<ButtonRelease-1>')
        self.plan_num_label.grid_forget()
        # self.a_slider.grid_forget()
        # self.a_slider.unbind('<ButtonRelease-1>')
        # self.a_value.grid_forget()
        self.rerun_button.grid_forget()
        self.end_program_button.grid_forget()
        self.restart_button.grid_forget()
        self.buttons_frame.grid_forget()
        self.optimization()

    def restart(self):
        self.final_canvas.destroy()
        self.final_canvas_ad.destroy()
        #self.final_canvas_orig.destroy()
        self.final_canvas_frame.grid_forget()
        self.plan_num_slider.grid_forget()
        self.plan_num_slider.unbind('<ButtonRelease-1>')
        self.plan_num_label.grid_forget()
        self.max_label.grid_forget()
        self.min_label.grid_forget()
        # self.a_slider.grid_forget()
        # self.a_slider.unbind('<ButtonRelease-1>')
        # self.a_value.grid_forget()
        # self.rerun_button.grid_forget()
        self.end_program_button.grid_forget()
        self.restart_button.grid_forget()
        self.buttons_frame.grid_forget()
        self.slider_frame.grid_forget()

        self.get_started_button.grid_forget()
        #self.suture_planner_text.configure(text='Welcome to Suture-It. Upload a wound image to start!')
        self.suture_planner_text.configure(text='Upload a wound image to start!')
        self.upload_inst.grid(row=4, column=0, padx=20, pady=10)
        self.example_fig.grid(row=5, column=0, padx=20, pady=0)
        self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)
        self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)

        # self.scale_pts = []
        # self.a = 0.5
        self.scale_pts = [(331,120),(342,160)]
        self.a = 0.77
    
    def change_suture_plan(self,event):
        self.final_canvas.delete('finalsutures')
        self.final_canvas_ad.delete('finalsutures')
        self.plan_num = math.floor(self.plan_num_slider.get())
        self.plan_num_slider.set(self.plan_num)
        self.plan_num_label.configure(text=f'To view other suture plans, adjust number of sutures using slider.\nDisplaying {self.plan_num} sutures.')

        # plot suture points
        in_pts = self.optFrame.planned_insert_pts[self.plan_num-self.start_range]
        cen_pts = self.optFrame.planned_center_pts[self.plan_num-self.start_range]
        ex_pts = self.optFrame.planned_extract_pts[self.plan_num-self.start_range]

        in_n_pts = self.optFrame.planned_n_insert_pts[self.plan_num-self.start_range]
        ex_n_pts = self.optFrame.planned_n_extract_pts[self.plan_num-self.start_range]

        r = 3
        print('\nDistances between sutures in suture plan:')
        for i in range(len(in_pts)):
            # draw centerline
            if i != 0: # create_line(x1, y1, x2, y2)
                self.final_canvas.create_line(cen_pts[i][0],cen_pts[i][1],cen_pts[i-1][0],cen_pts[i-1][1],fill='green',width=1,tags='finalsutures')
                self.final_canvas_ad.create_line(cen_pts[i][0],cen_pts[i][1],cen_pts[i-1][0],cen_pts[i-1][1],fill='green',width=1,tags='finalsutures')
                d = math.sqrt(((cen_pts[i-1][0] - cen_pts[i][0])**2) + ((cen_pts[i-1][1] - cen_pts[i][1])**2))
                print(str(i) + ': ' + str(d))

            # draw points and suture lines
            self.final_canvas.create_oval(in_pts[i][0]-r,in_pts[i][1]-r,in_pts[i][0]+r,in_pts[i][1]+r,fill='red',outline='',tags='finalsutures') #Insertion
            self.final_canvas.create_oval(ex_pts[i][0]-r,ex_pts[i][1]-r,ex_pts[i][0]+r,ex_pts[i][1]+r,fill='blue',outline='',tags='finalsutures') #Extraction
            #self.final_canvas.create_oval(self.optFrame.center_pts[i][0]-r,self.optFrame.center_pts[i][1]-r,self.optFrame.center_pts[i][0]+r,self.optFrame.center_pts[i][1]+r,fill='green') #Center
            self.final_canvas.create_line(in_pts[i][0],in_pts[i][1],ex_pts[i][0],ex_pts[i][1],fill='black',width=1.5,tags='finalsutures')

            self.final_canvas_ad.create_oval(in_n_pts[i][0]-r,in_n_pts[i][1]-r,in_n_pts[i][0]+r,in_n_pts[i][1]+r,fill='red',outline='',tags='finalsutures') #Insertion
            self.final_canvas_ad.create_oval(ex_n_pts[i][0]-r,ex_n_pts[i][1]-r,ex_n_pts[i][0]+r,ex_n_pts[i][1]+r,fill='blue',outline='',tags='finalsutures') #Extraction
            self.final_canvas_ad.create_line(in_n_pts[i][0],in_n_pts[i][1],ex_n_pts[i][0],ex_n_pts[i][1],fill='black',width=1.5,tags='finalsutures')

    def view_final(self):
        self.optFrame.grid_forget()
        self.see_final_opt.grid_forget()

        # place frame for both images
        self.final_canvas_frame.grid(row=4, column=0, padx=20, pady=20)

        # original color image for reference
        self.image = Image.open(self.image_path).resize((600,400))
        self.tk_image = ImageTk.PhotoImage(self.image)
        # self.final_canvas_orig = ctk.CTkCanvas(self.final_canvas_frame, width=600, height=400, bg='#f8f9fa', highlightthickness=0)
        # self.final_canvas_orig.grid(row=0, column=0, padx=5, pady=5)
        # self.final_canvas_orig.create_image(0,0,anchor=tk.NW,image=self.tk_image)

        # transparent image with suture plan
        self.final_canvas = ctk.CTkCanvas(self.final_canvas_frame, width=600, height=400, bg='#f8f9fa', highlightthickness=0)
        self.final_canvas.grid(row=0, column=1, padx=5, pady=0)

        # transparent image with adaptive suture plan
        self.final_canvas_ad = ctk.CTkCanvas(self.final_canvas_frame, width=600, height=400, bg='#f8f9fa', highlightthickness=0)
        self.final_canvas_ad.grid(row=0, column=0, padx=5, pady=0)

        self.final_canvas_cap = ctk.CTkLabel(self.final_canvas_frame, text='Uniform length sutures along centerline of simulated closed wound', font=('Arial',17), wraplength=600, text_color='#003049')
        self.final_canvas_ad_cap = ctk.CTkLabel(self.final_canvas_frame, text='Variable length sutures showing insert/extract points on open wound skin', font=('Arial',17), wraplength=600, text_color='#003049')
        self.final_canvas_cap.grid(row=1, column=1, padx=5, pady=0)
        self.final_canvas_ad_cap.grid(row=1, column=0, padx=5, pady=0)

        self.image = Image.open(self.image_path).resize((600,400))
        
        # make image more transparent to see sutures in final plan better
        self.image = self.image.convert('RGBA')
        alpha_factor = 0.6
        transparent_image_data = []
        for pixel in self.image.getdata():
            if pixel[3] > 0:
                new_alpha_factor = int(pixel[3] * alpha_factor)
                transparent_image_data.append((pixel[0],pixel[1],pixel[2],new_alpha_factor))
            else:
                # if transparents already, keep it transparent
                transparent_image_data.append(pixel)
        adjusted_image = Image.new('RGBA',self.image.size)
        adjusted_image.putdata(transparent_image_data)
        #self.tk_image = ImageTk.PhotoImage(self.image)
        self.tk_trans_image = ImageTk.PhotoImage(adjusted_image)

        self.final_canvas.create_image(0,0,anchor=tk.NW,image=self.tk_trans_image)
        self.final_canvas.image = self.tk_trans_image # keep reference to image

        self.final_canvas_ad.create_image(0,0,anchor=tk.NW,image=self.tk_trans_image)
        self.final_canvas_ad.image = self.tk_trans_image # keep reference to image

        # plot suture points
        r = 3
        for i in range(self.optFrame.num_pts):
            # draw centerline
            if i != 0:
                self.final_canvas.create_line(self.optFrame.center_pts[i][0],self.optFrame.center_pts[i][1],self.optFrame.center_pts[i-1][0],self.optFrame.center_pts[i-1][1],fill='green',width=1.5,tags='finalsutures')
                self.final_canvas_ad.create_line(self.optFrame.center_pts[i][0],self.optFrame.center_pts[i][1],self.optFrame.center_pts[i-1][0],self.optFrame.center_pts[i-1][1],fill='green',width=1.5,tags='finalsutures')

            # draw points and suture lines
            self.final_canvas.create_oval(self.optFrame.insert_pts[i][0]-r,self.optFrame.insert_pts[i][1]-r,self.optFrame.insert_pts[i][0]+r,self.optFrame.insert_pts[i][1]+r,fill='red',outline='',tags='finalsutures') #Insertion
            self.final_canvas.create_oval(self.optFrame.extract_pts[i][0]-r,self.optFrame.extract_pts[i][1]-r,self.optFrame.extract_pts[i][0]+r,self.optFrame.extract_pts[i][1]+r,fill='blue',outline='',tags='finalsutures') #Extraction
            #self.final_canvas.create_oval(self.optFrame.center_pts[i][0]-r,self.optFrame.center_pts[i][1]-r,self.optFrame.center_pts[i][0]+r,self.optFrame.center_pts[i][1]+r,fill='green') #Center
            #scale = 1
            #if self.wound_mask[self.optFrame.insert_pts[i][1],self.optFrame.insert_pts[i][0]] == 255:
            self.final_canvas.create_line(self.optFrame.insert_pts[i][0],self.optFrame.insert_pts[i][1],self.optFrame.extract_pts[i][0],self.optFrame.extract_pts[i][1],fill='black',width=1.5,tags='finalsutures')

            # adpative length
            self.final_canvas_ad.create_oval(self.optFrame.n_insert_pts[i][0]-r,self.optFrame.n_insert_pts[i][1]-r,self.optFrame.n_insert_pts[i][0]+r,self.optFrame.n_insert_pts[i][1]+r,fill='red',outline='',tags='finalsutures') #Insertion
            self.final_canvas_ad.create_oval(self.optFrame.n_extract_pts[i][0]-r,self.optFrame.n_extract_pts[i][1]-r,self.optFrame.n_extract_pts[i][0]+r,self.optFrame.n_extract_pts[i][1]+r,fill='blue',outline='',tags='finalsutures') #Extraction
            self.final_canvas_ad.create_line(self.optFrame.n_insert_pts[i][0],self.optFrame.n_insert_pts[i][1],self.optFrame.n_extract_pts[i][0],self.optFrame.n_extract_pts[i][1],fill='black',width=1.5,tags='finalsutures')
            

        self.suture_planner_text.configure(text='- Optimized Suture Placement Plan -\nWound skin is pulled together along centerline before suturing.')
        self.buttons_frame.grid(row=100,column=0,padx=20,pady=5)
        self.slider_frame.grid(row=1,column=0,padx=10,pady=5)

        self.plan_num_slider = ctk.CTkSlider(self.slider_frame,from_=self.start_range, to=self.end_range-1,orientation='horizontal',width=250)
        self.plan_num_slider.set(self.optFrame.best_sutures)
        self.plan_num_label = ctk.CTkLabel(self.buttons_frame, text=f'To view other suture plans, adjust number of sutures using slider.\nDisplaying {self.optFrame.best_sutures} sutures.',font=('Arial',17), text_color='#003049')
        self.min_label = ctk.CTkLabel(self.slider_frame, text=f'{self.start_range}',font=('Arial',17), text_color='#003049')
        self.max_label = ctk.CTkLabel(self.slider_frame, text=f'{self.end_range-1}',font=('Arial',17), text_color='#003049')

        self.plan_num_label.grid(row=0, column=0, padx=10,pady=0)
        self.plan_num_slider.grid(row=0,column=1,padx=0,pady=0)
        self.plan_num_slider.bind('<ButtonRelease-1>',self.change_suture_plan)
        self.min_label.grid(row=0,column=0,padx=0,pady=0)
        self.max_label.grid(row=0,column=2,padx=0,pady=0)
        # self.a_value.grid(row=0, column=0, padx=0,pady=10)
        # self.a_slider.grid(row=1,column=0,padx=0,pady=10)
        # self.a_slider.bind('<ButtonRelease-1>',self.slider_update)
        # self.rerun_button.grid(row=0, column=1, padx=20, pady=10)
        self.restart_button.grid(row=0, column=1, padx=40, pady=5)
        self.end_program_button.grid(row=1, column=1, padx=40, pady=0)

if __name__ == '__main__':
    app = GUI()
    app.configure(fg_color="#f8f9fa")
    app.mainloop()