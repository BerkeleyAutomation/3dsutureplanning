import customtkinter as ctk
from tkinter import filedialog
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
import matplotlib.pyplot as plt

from InsertionPointGenerator import InsertionPointGenerator
from ScaleGenerator import ScaleGenerator
from SutureDisplayAdjust2D import SutureDisplayAdjust2D
import numpy as np
import cv2
import math
from PIL import Image, ImageTk, ImageDraw

import tkinter as tk

import EdgeDetector
from PIL import Image


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("green")

        self.title("Suture Planning App")
        self.geometry("750x750")
        self.grid_columnconfigure(0, weight=1)

        suture_planner_label = ctk.CTkLabel(self, text="Suture Planner", font=("Arial Bold", 40), fg_color="#2fc986", text_color="white")
        suture_planner_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.suture_planner_text = ctk.CTkLabel(self, text="Welcome to the Suture Planner. Upload an image of a wound to get started!", font=("Arial", 15), wraplength=400)
        self.suture_planner_text.grid(row=1, column=0, padx=20, pady=20)

        #self.exit_button = ctk.CTkButton(self, text="X", command=self.destroy, width=50, height=50, font=("Arial", 17))
        #self.exit_button.grid(row=0, column=100, padx=20, pady=20)


        self.upload_image_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image, width=150, height=50, font=("Arial", 17))
        self.upload_image_button.grid(row=100, column=0, padx=20, pady=20)

        self.done_clicking = ctk.CTkButton(self, text="Done!", command=self.done_clicking, width=150, height=50, font=("Arial", 17))
        self.generate_mask = ctk.CTkButton(self, text="Compute Centerline", command=self.compute_centerline, width=150, height=50, font=("Arial", 17))
        self.start_opt = ctk.CTkButton(self, text="Start Optimization", command=self.optimization, width=150, height=50, font=("Arial", 17))


        self.scale_pts = []

    def start_draw(self, event):
        if hasattr(self, "line_ids"):
            for line_id in self.line_ids:
                self.canvas.delete(line_id)
        self.points = [(event.x, event.y)]
        self.line_ids = []
        self.drawing = True
        

    def draw(self, event):
        if self.drawing:
            self.points.append((event.x, event.y))
            if len(self.points) >= 2:
                line_id = self.canvas.create_line(*self.points[-2], *self.points[-1], fill='blue', width=3)
                self.line_ids.append(line_id)
    def end_draw(self, event):
        self.drawing = False
        self.points.append((event.x, event.y))

        if len(self.points) > 2:
            # Close polygon with a line
            line_id = self.canvas.create_line(*self.points[-1], *self.points[0], fill='blue', width=3)
            self.line_ids.append(line_id)

            h, w = self.tk_image.height(), self.tk_image.width()
            self.mask = np.zeros((h, w), dtype=np.uint8)

            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], 255)

            base_image = self.image.convert("RGBA")

            overlay = Image.new("RGBA", (w, h), (0, 0, 255, 0))
            for y in range(h):
                for x in range(w):
                    if self.mask[y, x] == 255:
                        overlay.putpixel((x, y), (0, 0, 255, 100))  # semi-transparent blue

            blended = Image.alpha_composite(base_image, overlay)

            self.tk_image = ImageTk.PhotoImage(blended)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.image = self.tk_image  # keep a reference


    def optimization(self):
        print('Starting Optimization')
        # self.start_opt.grid_forget()
        self.canvas.destroy()
        
        self.suture_planner_text.configure(text="Suture Optimization Process Initiated! Please wait...")

        print('Creating Progress Bar')

        def updateprogress(progressbar, cur_val, tar_val, step, ms):
            if cur_val < tar_val:
                new_val = min(cur_val + step, tar_val)
                progressbar.set(new_val)
                progressbar.master.after(ms, updateprogress, progressbar, new_val, tar_val, step, ms)

        self.progress_bar = ctk.CTkProgressBar(self, orientation='horizontal', mode='determinate', width=300)
        self.progress_bar.grid(row=50, column=0, padx=20, pady=20)
        self.progress_bar.set(0)
        updateprogress(self.progress_bar,0,1,0.01,100)
        #self.progress_bar.start()


    def compute_centerline(self):
        ordered_points, _, _ = EdgeDetector.img_to_line(self.image_path, self.mask)

        x = [a[1] for a in ordered_points]
        y = [a[0] for a in ordered_points]



        base_image = self.image.convert("RGB")
        draw = ImageDraw.Draw(base_image)

        for pt in ordered_points:
            draw.ellipse((pt[1] - 2, pt[0] - 2, pt[1] + 2, pt[0] + 2), fill="red")

        self.tk_image = ImageTk.PhotoImage(base_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image  # retain reference
        

        tck, u = inter.splprep([x, y], k=5)

        ## 

        print('-- Centerline Drawn -- ')

        self.generate_mask.grid_forget()
        self.start_opt.grid(row=100, column=0, padx=20, pady=20)
        self.suture_planner_text.configure(text="Wound centerline is displayed in red. Press the button below to begin suture placement optimization.")

        ## temp comment start

        # pixel_dist = math.sqrt((self.scale_pts[0][0] - self.scale_pts[1][0]) ** 2 + (self.scale_pts[0][1] - self.scale_pts[1][1]) ** 2)
        # mm_per_pixel = real_dist / pixel_dist
        # deg = 5



        # wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

        # # Put the wound into all the relevant objects
        # newSuturePlacer = SuturePlacer(5, mm_per_pixel)
        # newSuturePlacer.tck = tck
        # newSuturePlacer.DistanceCalculator.tck = tck

        # newSuturePlacer.wound_parametric = wound_parametric
        # newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        # newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        # newSuturePlacer.image = self.image_path

        
        # # The main algorithm
        # newSuturePlacer.place_sutures()
        # return newSuturePlacer

        ## temp comment end

        return
    
    
    def done_clicking(self):
        self.done_clicking.grid_forget()
        self.canvas.unbind("<Button-1>")
        self.canvas.delete("suture")
        
        self.suture_planner_text.configure(text="Now click and drag a region around the wound for us to segment!")


        self.drawing = False
        self.points = []
        self.line_ids = []

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        self.generate_mask.grid(row=100, column=0, padx=20, pady=20)
        
        # ordered_points, _, _ = EdgeDetector.img_to_line(self.image_path)


        # # pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
        # x = [a[1] for a in ordered_points]
        # y = [a[0] for a in ordered_points]

        # # # now, use our conversion factor to scale points appropriately
        # x = [float(elem) * mm_per_pixel for elem in x]
        # y = [float(elem) * -mm_per_pixel for elem in y]
        # deg = 5

        # tck, u = inter.splprep([x, y], k=deg)
        # wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

        # # progress_gui = start_progress_gui("Suture Planning Progress")


        # # Put the wound into all the relevant objects
        # newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
        # newSuturePlacer.tck = tck
        # newSuturePlacer.DistanceCalculator.tck = tck

        # newSuturePlacer.wound_parametric = wound_parametric
        # newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        # newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        # newSuturePlacer.image = self.image_path
        
        # # The main algorithm
        # newSuturePlacer.place_sutures()
        # return newSuturePlacer
        

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        # print(f"Clicked at: {x}, {y}")

        if len(self.scale_pts) >= 2:
            self.scale_pts = []
            self.canvas.delete("suture")

        self.scale_pts.append((x, y))
        r = 4
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="", tags="suture")

        if len(self.scale_pts) == 2:
            self.canvas.create_line(self.scale_pts[0][0], self.scale_pts[0][1],
                                    self.scale_pts[1][0], self.scale_pts[1][1],
                                    fill="black", width=2, tags="suture")

    def upload_image(self):
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        if image_path:
            self.image_path = image_path
            self.upload_image_button.grid_forget()
            
            self.image = Image.open(self.image_path).resize((600, 400))
            self.tk_image = ImageTk.PhotoImage(self.image)

            self.canvas = ctk.CTkCanvas(self, width=600, height=400)
            self.canvas.grid(row=4, column=0, padx=20, pady=20)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            self.suture_planner_text.configure(text="Now draw an example suture on the image to help us estimate suture width by clicking the entry and exit points one at a time. ")

            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.done_clicking.grid(row=100, column=0, padx=20, pady=20)
        else:
            print("No image selected. Exiting.")
            return
        


    def suture_placing_pipeline(self, image_path):

        # display the image on the GUI and make the user click 2 points on the image
        

        # make a new scale object to get the scale
        newScale = ScaleGenerator()
        space_between_sutures = 0.010  # 1 cm
        desired_compute_time = 1
        IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                    space_between_sutures=space_between_sutures)

        img_color = cv2.imread(image_path)
        img_point = np.load("record/img_point_inclined.npy")

        # get the scale measurement from surgeon (this shows the draw mask dialog)
        scale_pts = newScale.get_scale_pts(img_color, img_point)

        # request the surgeon for a distance
        real_dist = 5
        # real_dist = simpledialog.askfloat(title="dist prompt",
        #                                   prompt="Please enter the distance in mm that you measured")

        # wound_width = simpledialog.askfloat(title="width prompt", prompt="Please enter the width of suture in mm (insertion to center)")
        wound_width = 5
        cv2.destroyAllWindows()
        
        # progress_gui = start_progress_gui("Suture Planning Progress")
        
        # Give GUI time to start up
        # time.sleep(0.5)

        pixel_dist = math.sqrt((scale_pts[0][0] - scale_pts[1][0]) ** 2 + (scale_pts[0][1] - scale_pts[1][1]) ** 2)
        real_dist = 5
        mm_per_pixel = real_dist / pixel_dist
        deg = 5
        
        ordered_points, _, _ = EdgeDetector.img_to_line(image_path)


        # pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
        x = [a[1] for a in ordered_points]
        y = [a[0] for a in ordered_points]

        # # now, use our conversion factor to scale points appropriately
        x = [float(elem) * mm_per_pixel for elem in x]
        y = [float(elem) * -mm_per_pixel for elem in y]
        deg = 5

        tck, u = inter.splprep([x, y], k=deg)
        wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

        # progress_gui = start_progress_gui("Suture Planning Progress")


        # Put the wound into all the relevant objects
        newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
        newSuturePlacer.tck = tck
        newSuturePlacer.DistanceCalculator.tck = tck

        newSuturePlacer.wound_parametric = wound_parametric
        newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        newSuturePlacer.image = image_path
        
        # The main algorithm
        newSuturePlacer.place_sutures()
        return newSuturePlacer

    def suture_display_adj_pipeline(newSuturePlacer):
        insert_pts = newSuturePlacer.b_insert_pts
        center_pts = newSuturePlacer.b_center_pts
        extract_pts = newSuturePlacer.b_extract_pts
        mm_per_pixel = newSuturePlacer.mm_per_pixel

        newSutureDisAdj = SutureDisplayAdjust2D(insert_pts, center_pts, extract_pts, mm_per_pixel)
        
        # display
        img_color = cv2.imread(newSuturePlacer.image)
        img_point = np.load("record/img_point_inclined.npy")

        # allow for edit
        newSutureDisAdj.adjust_points(img_color, img_point)
        return



if __name__ == "__main__":
    app = GUI()
    app.mainloop()
