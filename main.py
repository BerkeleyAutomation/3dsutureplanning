# Import Packages
import customtkinter as ctk
from tkinter import filedialog
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# from InsertionPointGenerator import InsertionPointGenerator
# from ScaleGenerator import ScaleGenerator
# from SutureDisplayAdjust2D import SutureDisplayAdjust2D
# import RewardFunction
import numpy as np
import cv2
import math
from PIL import Image, ImageTk, ImageDraw

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
        self.best_num_sutures = 0
        
        self.cur_visualization_data = None
        self.distance_calc = None
        self.parent_root = parent

        # LEFT SIDE
        # frame for optimization details
        self.infopanel = ctk.CTkFrame(self)
        self.infopanel.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.infopanel.grid_rowconfigure(0,weight=1)
        self.infopanel.grid_columnconfigure(0,weight=1)
        self.infopanel.grid_columnconfigure(2,weight=1)

        infoheader = ctk.CTkLabel(self.infopanel, text='Optimization Progress Details', font=('Arial',17,'bold'),justify='center')
        infoheader.grid(row=0,column=1,padx=10,pady=10,sticky='ew')
        progress_label = ctk.CTkLabel(self.infopanel, text='Current Progress', font=('Arial',17),justify='center')
        progress_label.grid(row=1,column=1,padx=10,pady=10,sticky='ew')

        # left panel progress bar
        self.progress_bar = ctk.CTkProgressBar(self.infopanel, orientation='horizontal', mode='determinate', width=300)
        self.progress_bar.grid(row=2,column=1,padx=10,pady=10)
        self.progress_bar.set(0)
        self.progress_percent = ctk.CTkLabel(self.infopanel,text='0%',font=('Arial',12),justify='center')
        self.progress_percent.grid(row=3,column=1,padx=5,pady=5)

        self.cur_suture_info = ctk.CTkLabel(self.infopanel, text='Testing: 0 sutures', font=('Arial',12),justify='center')
        self.cur_suture_info.grid(row=4,column=1,padx=5,pady=5)

        # loss information
        self.loss_frame = ctk.CTkFrame(self.infopanel)
        self.loss_frame.grid(row=5,column=1,padx=10,pady=10,sticky='ew')

        self.loss_frame.grid_rowconfigure(0,weight=1)
        self.loss_frame.grid_columnconfigure(0,weight=1)
        self.loss_frame.grid_columnconfigure(2,weight=1)

        self.total_loss_label = ctk.CTkLabel(self.loss_frame, text='Total Loss: --', font=('Arial', 12),justify='center')
        self.total_loss_label.grid(row=0,column=1,padx=10,sticky='ew')
        self.closure_loss_label = ctk.CTkLabel(self.loss_frame, text='Closure Loss: --', font=('Arial',12),justify='center')
        self.closure_loss_label.grid(row=1,column=1,padx=10,sticky='ew')
        self.shear_loss_label = ctk.CTkLabel(self.loss_frame, text='Shear Loss: --', font=('Arial',12),justify='center')
        self.shear_loss_label.grid(row=2,column=1,padx=10,sticky='ew')
        
        # best result information
        self.best_frame = ctk.CTkFrame(self.infopanel,fg_color='#2fc986') #'#2fc986'
        self.best_frame.grid(row=6,column=1,padx=10,pady=10,sticky='ew')

        self.best_frame.grid_rowconfigure(0,weight=1)
        self.best_frame.grid_columnconfigure(0,weight=1)
        self.best_frame.grid_columnconfigure(2,weight=1)

        self.best_label = ctk.CTkLabel(self.best_frame, text='Best Result: --', font=('Arial',12),justify='center',text_color='white')
        self.best_label.grid(row=0,column=1,padx=10,pady=10,sticky='ew')

        # RIGHT SIDE
        # frame for optimization graph visualization
        self.graphpanel = ctk.CTkFrame(self)
        self.graphpanel.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.graph_title = ctk.CTkLabel(self.graphpanel, text='Current Suture Plan Visualization', font=('Arial',17,'bold'))
        self.graph_title.grid(row=0,column=0,padx=10,pady=10)

        self.graph_fig = Figure(figsize=(4,4), dpi=80)
        self.graph_ax = self.graph_fig.add_subplot(111)
        
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, self.graphpanel)
        self.graph_canvas.get_tk_widget().grid(row=1,column=0,padx=10,pady=10)

        # inital plot
        self.graph_ax.text(0.5,0.5,'Waiting for optimization...',ha='center',va='center',transform=self.graph_ax.transAxes,fontsize=12)
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

        self.cur_suture_info.configure(text=f'Testing: {num_sutures} sutures')
        if self.test_suture_range[1] > 0:
            progress = (num_sutures - self.test_suture_range[0]) / (self.test_suture_range[1] - self.test_suture_range[0] + 1)
            self.update_progress(progress)
    
    def update_losses(self, total_loss: float, closure_loss: float, shear_loss: float):
        # update display of loss information
        self.total_loss_label.configure(text=f'Total Loss: {total_loss:.4f}')
        self.closure_loss_label.configure(text=f'Closure Loss: {closure_loss:.4f}')
        self.shear_loss_label.configure(text=f'Shear Loss: {shear_loss:.4f}')

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_sutures = self.cur_num_sutures
            self.best_label.configure(text=f'Best Result: {self.best_sutures} sutures - Loss: {self.best_loss:.4f}')

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
        
        # plot suture points
        self.graph_ax.scatter([insert_pts[i][1] for i in range(num_pts)],[-insert_pts[i][0] for i in range(num_pts)],c='red',s=30,alpha=0.8,label='Insertion')
        self.graph_ax.scatter([extract_pts[i][1] for i in range(num_pts)],[-extract_pts[i][0] for i in range(num_pts)],c='blue',s=30,alpha=0.8,label='Extraction')
        #self.graph_ax.scatter([center_pts[i][1] for i in range(num_pts)],[-center_pts[i][0] for i in range(num_pts)],c='green',s=30,alpha=0.8,label='Center')

        # draw suture lines on plot
        for i in range(len(insert_pts)):
            self.graph_ax.plot([insert_pts[i][1],extract_pts[i][1]],[-insert_pts[i][0],-extract_pts[i][0]],color='black',linewidth=1,alpha=0.6)
        
        self.graph_ax.set_title(f'{title}\n{self.cur_num_sutures} sutures - Loss: {self.best_loss:.4f}', fontsize=12)
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
        print(f'Best results: {self.best_sutures} sutures with {self.best_loss:.4f} loss')

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('green')

        self.title('Suture Planning App')
        self.geometry('1000x800') #'750x750'
        self.grid_columnconfigure(0, weight=1)

        suture_planner_title = ctk.CTkLabel(self, text='Suture Planner', font=('Arial Bold',40), fg_color='#2fc986', text_color='white')
        suture_planner_title.grid(row=0, column=0, padx=20, pady=20, sticky='ew')

        self.suture_planner_text = ctk.CTkLabel(self, text='Welcome to the Suture Planner. Upload an image of a wound to get started!', font=('Arial',17), wraplength=400)
        self.suture_planner_text.grid(row=1, column=0, padx=20, pady=30)

        self.upload_image_button = ctk.CTkButton(self, text='Upload Image', command=self.upload_image, width=150, height=50, font=('Arial',17))
        self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)

        self.scale_pts = []
        self.a = 0.5

        self.suture_drawn_button = ctk.CTkButton(self, text='Done!', command=self.suture_drawn, width=150, height=50, font=('Arial',17))
        self.mask_generated = ctk.CTkButton(self, text='Compute Wound Centerline', command=self.compute_centerline, width=150, height=50, font=('Arial',17))
        self.start_opt = ctk.CTkButton(self, text='Start Optimization', command=self.optimization, width=150, height=50, font=('Arial',17))
        self.see_final_opt = ctk.CTkButton(self, text='View Optimized Suture Plan', command=self.view_final, width=150, height=50, font=('Arial',17))
        
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.grid_rowconfigure(0,weight=1)

        self.a_slider = ctk.CTkSlider(self.buttons_frame,from_=0, to=1,orientation='horizontal',width=150)
        self.a_value = ctk.CTkLabel(self.buttons_frame, text=f'Elliptical Minor Axis = {round(self.a,2)}\nIncreasing will relax distance between sutures.',font=('Arial',12))
        self.rerun_button = ctk.CTkButton(self.buttons_frame, text='Rerun Program', command=self.rerun, width=150, height=50, font=('Arial',17))
        self.end_program_button = ctk.CTkButton(self.buttons_frame, text='Close Program', command=self.on_close, width=150, height=50,font=('Arial',17))

    def on_close(self):
        #self.destroy()
        self.quit()
        sys.exit()
    
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
    
    def start_draw(self, event):
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
        self.suture_drawn_button.grid_forget()
        self.image_canvas.unbind('<Button-1>')
        self.image_canvas.delete('suture')

        self.suture_planner_text.configure(text='Click and drag to draw the outline of the wound! Release when done. The region should be fully enclosed. Click and drag again to redraw if needed.')

        self.drawing = False
        self.points = []
        self.line_ids = []

        self.image_canvas.bind('<ButtonPress-1>', self.start_draw)
        self.image_canvas.bind('<B1-Motion>', self.draw)
        self.image_canvas.bind('<ButtonRelease-1>', self.end_draw)

        self.mask_generated.grid(row=100, column=0, padx=20, pady=20)

    def upload_image(self):
        image_path = filedialog.askopenfilename(title='Select an Image', filetypes=[('Image Files','*.jpg *.jpeg *.png *.bmp *.tiff *.gif')])
        
        if image_path:
            self.image_path = image_path
            self.upload_image_button.grid_forget()
            self.image = Image.open(self.image_path).resize((600,400))
            self.tk_image = ImageTk.PhotoImage(self.image)

            self.image_canvas = ctk.CTkCanvas(self, width=600, height=400)
            self.image_canvas.grid(row=4, column=0, padx=20, pady=20)
            self.image_canvas.create_image(0,0,anchor=tk.NW,image=self.tk_image)

            self.suture_planner_text.configure(text='Draw an example suture on the image by clicking two endpoints. The example suture should have your desired estimate suture length.')
            
            self.image_canvas.bind('<Button-1>', self.on_image_click)
            self.suture_drawn_button.grid(row=100, column=0, padx=20, pady=20)
        else:
            print('No image selected. Exiting.')
            return
    
    def compute_centerline(self):
        ordered_pts, _, _ = EdgeDetector.img_to_line(self.image_path, self.wound_mask)

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
        self.suture_planner_text.configure(text='Wound centerline is displayed in red. Press the button below to begin suture placement optimization.')
        self.start_opt.grid(row=100, column=0, padx=20, pady=20)
    
    def optimization(self):
        print('Starting Optimization')
        self.start_opt.grid_forget()
        self.image_canvas.destroy()
        self.suture_planner_text.configure(text='Running Suture Placement Optimization!')

        # Start progress canvas (progress bar and update information)
        # Create instance of frame for optimization progress
        self.optFrame = OptFrame(parent=self, width=500, height=500, border_width=2, border_color='green', fg_color='transparent')
        self.optFrame.grid(row=50, column=0, padx=20, pady=20, sticky='nsew')

        # main optimization algorithm
        print('Starting Main Algorithm')
        wound_width = 5
        real_dist = 5
        pixel_dist = math.sqrt((self.scale_pts[0][0] - self.scale_pts[1][0])**2 + (self.scale_pts[0][1] - self.scale_pts[1][1])**2)
        mm_per_pixel = real_dist / pixel_dist
        wound_parametric = lambda t,d: inter.splev(t,self.tck,der=d)

        newSuturePlacer = SuturePlacer(wound_width,mm_per_pixel)
        newSuturePlacer.tck = self.tck
        newSuturePlacer.DistanceCalculator.tck = self.tck
        newSuturePlacer.RewardFunction.a = self.a
        self.a_value.configure(text=f'Elliptical Minor Axis = {round(self.a,2)}\nIncreasing will relax distance between sutures.')

        newSuturePlacer.wound_parametric = wound_parametric
        newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        newSuturePlacer.image = self.image_path
        newSuturePlacer.place_sutures(_optFrame=self.optFrame)

        self.see_final_opt.grid(row=100, column=0, padx=20, pady=20)
    
    def rerun(self):
        self.a = self.a_slider.get()
        print(f'Rerunning program with a = {round(self.a,2)}')
        self.final_canvas.destroy()
        self.a_slider.grid_forget()
        self.a_slider.unbind('<ButtonRelease-1>')
        self.a_value.grid_forget()
        self.rerun_button.grid_forget()
        self.end_program_button.grid_forget()
        self.buttons_frame.grid_forget()
        self.optimization()
    
    def view_final(self):
        self.optFrame.grid_forget()
        self.see_final_opt.grid_forget()

        self.final_canvas = ctk.CTkCanvas(self, width=600, height=400)
        self.final_canvas.grid(row=4, column=0, padx=20, pady=20)
        
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

        # plot suture points
        r = 3
        for i in range(self.optFrame.num_pts):
            # draw centerline
            if i != 0:
                self.final_canvas.create_line(self.optFrame.center_pts[i][0],self.optFrame.center_pts[i][1],self.optFrame.center_pts[i-1][0],self.optFrame.center_pts[i-1][1],fill='green',width=1)

            # draw points and suture lines
            self.final_canvas.create_oval(self.optFrame.insert_pts[i][0]-r,self.optFrame.insert_pts[i][1]-r,self.optFrame.insert_pts[i][0]+r,self.optFrame.insert_pts[i][1]+r,fill='red',outline='') #Insertion
            self.final_canvas.create_oval(self.optFrame.extract_pts[i][0]-r,self.optFrame.extract_pts[i][1]-r,self.optFrame.extract_pts[i][0]+r,self.optFrame.extract_pts[i][1]+r,fill='blue',outline='') #Extraction
            #self.final_canvas.create_oval(self.optFrame.center_pts[i][0]-r,self.optFrame.center_pts[i][1]-r,self.optFrame.center_pts[i][0]+r,self.optFrame.center_pts[i][1]+r,fill='green') #Center
            self.final_canvas.create_line(self.optFrame.insert_pts[i][0],self.optFrame.insert_pts[i][1],self.optFrame.extract_pts[i][0],self.optFrame.extract_pts[i][1],fill='black',width=1.5)
        
        self.suture_planner_text.configure(text='- Final Optimized Suture Placement Plan -\nWe assume the wound skin is pulled together along the wound centerline during suturing.')
        self.buttons_frame.grid(row=100,column=0,padx=20,pady=10)
        self.a_slider.grid(row=0,column=0,padx=0,pady=10)
        self.a_slider.bind('<ButtonRelease-1>',self.slider_update)
        self.a_value.grid(row=0, column=1, padx=0,pady=10)
        self.rerun_button.grid(row=1, column=0, padx=20, pady=10)
        self.end_program_button.grid(row=1, column=1, padx=20, pady=10)

if __name__ == '__main__':
    app = GUI()
    app.mainloop()