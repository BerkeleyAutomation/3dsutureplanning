import customtkinter as ctk
from tkinter import filedialog
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
from InsertionPointGenerator import InsertionPointGenerator
from ScaleGenerator import ScaleGenerator
from SutureDisplayAdjust2D import SutureDisplayAdjust2D
import numpy as np
import cv2
import math
import EdgeDetector
from PIL import Image


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("#2fc986")

        self.title("Suture Planning App")
        self.geometry("700x450")
        self.grid_columnconfigure((0, 1), weight=1)

        self.upload_image_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_image_button.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)


    def upload_image(self):
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        if image_path:
            print(f"Selected image: {image_path}")
            self.upload_image_button.place_forget()
            self.image = ctk.CTkImage(Image.open(image_path), size=(500, 150))
            self.image_label = ctk.CTkLabel(self, text="", image=self.image)
            # self.image_label.grid(row=0, column=0, padx=20, pady=10)
            # suturePlacerTest = self.suture_placing_pipeline(image_path)

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