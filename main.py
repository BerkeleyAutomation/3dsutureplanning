import scipy_generate_sample_spline
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
from InsertionPointGenerator import InsertionPointGenerator
from ScaleGenerator import ScaleGenerator
from SutureDisplayAdjust2D import SutureDisplayAdjust2D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import EdgeDetector
import tkinter as tk
from tkinter import simpledialog
import sys
import time
import threading
from progress_gui import start_progress_gui, stop_progress_gui
from image_upload_gui import upload_image_gui

NUM_EXAMPLES = 5

def suture_placing_pipeline(sample_spline=None, image=None):
    try:
        if sample_spline is None:

            # make a new scale object to get the scale
            newScale = ScaleGenerator()
            space_between_sutures = 0.010  # 1 cm
            desired_compute_time = 1
            IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                          space_between_sutures=space_between_sutures)

            img_color = cv2.imread(image)
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
        else:
            # Start progress GUI for pre-saved wound case
            progress_gui = start_progress_gui("Suture Planning Progress")
            
            # Give GUI time to start up
            time.sleep(0.5)
            
            mm_per_pixel=1
            point_data = (
                [[[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]], 
                 [[0, 3, 10, 15, 21, 25, 30], [0, 1, 3, 5, 3, 1, 0]],  
                 [[0, 7, 10, 15, 21, 25, 30], [0, -5, 5, 35, 18, 7, 13]],
                 [[0, 3, 10, 15, 10, 3, 1], [0, 1, 3, 5, 8, 10, 5]],
                 [[0, 3, 10, 18, 29, 35, 40], [0, -10, 5, -20, 18, 7, 13]]
                 ])
            
            width_data = [1.5, 1.5, 1.5, 1.5, 1.25]
            degree_data = [5, 2, 5, 5, 5]

            if 0 <= sample_spline < NUM_EXAMPLES:
                pnts = point_data[sample_spline]
                wound_width = width_data[sample_spline]
                deg = degree_data[sample_spline]

            else:
                raise Exception("not a pre-saved wound")
            
        
        ordered_points, _, _ = EdgeDetector.img_to_line(image)


        # pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
        x = [a[1] for a in ordered_points]
        y = [a[0] for a in ordered_points]

        # # now, use our conversion factor to scale points appropriately
        x = [float(elem) * mm_per_pixel for elem in x]
        y = [float(elem) * -mm_per_pixel for elem in y]
        deg = 5

        tck, u = inter.splprep([x, y], k=deg)
        wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

        progress_gui = start_progress_gui("Suture Planning Progress")


        # Put the wound into all the relevant objects
        newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
        newSuturePlacer.tck = tck
        newSuturePlacer.DistanceCalculator.tck = tck

        newSuturePlacer.wound_parametric = wound_parametric
        newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

        newSuturePlacer.image = image
        
        # The main algorithm
        newSuturePlacer.place_sutures(sample_spline)
        return newSuturePlacer
        
    finally:
        # Clean up progress GUI
        stop_progress_gui()

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

def main():
    """Main function with GUI support"""
    args = sys.argv
    
    # Initialize tkinter root
    ROOT = tk.Tk()
    ROOT.withdraw()  # Hide the main window

    if len(args) == 1:
        # No arguments provided - show image upload GUI
        print("No image specified. Opening image upload GUI...")
        image_path = upload_image_gui(parent_root=ROOT)
        
        if image_path:
            print(f"Selected image: {image_path}")
            suturePlacerTest = suture_placing_pipeline(sample_spline=None, image=image_path)
            suture_display_adj_pipeline(suturePlacerTest)
        else:
            print("No image selected. Exiting.")
            return
    
    elif len(args) == 3:
        # Arguments provided - use command line mode
        if args[1] == "-s":
            suturePlacerTest = suture_placing_pipeline(sample_spline=int(args[2]))
        elif args[1] == "-i":
            suturePlacerTest = suture_placing_pipeline(sample_spline=None, image=args[2])
            suture_display_adj_pipeline(suturePlacerTest)
        else:
            raise Exception("incorrect format: format is python main.py -i [image.jpg] || -s [spline_number]")
    
    else:
        raise Exception("incorrect format: format is python main.py -i [image.jpg] || -s [spline_number] || python main.py (for GUI)")
    
    cv2.destroyAllWindows()
    ROOT.destroy()  # Clean up the root window

if __name__ == "__main__":
    main()
