import numpy as np
import torch
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

def create_mask(imgPath, progress_tracker=None):
    # This function lets the user draw the outline of the mask with a pen tool,
    # and then fills in a closed concave mask.
    img = cv2.imread(imgPath)
    if img is None:
        raise ValueError("Could not load image from " + imgPath)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    drawing = False
    points = []
    
    # Create a copy of the original image for display
    display_img = img.copy()
    
    # Create a pen cursor
    pen_cursor = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(pen_cursor, (10, 10), 5, 255, -1)
    pen_cursor = cv2.cvtColor(pen_cursor, cv2.COLOR_GRAY2BGR)
    
    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, mask, points, display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
            # Reset display image to original
            display_img = img.copy()
            # Redraw instructions
            add_instructions(display_img)
            # Add progress bar if tracker is provided
            if progress_tracker:
                add_progress_bar_to_image(display_img, progress_tracker)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                points.append((x, y))
                # Reset display image to original
                display_img = img.copy()
                # Redraw instructions
                add_instructions(display_img)
                # Draw the current line
                for i in range(len(points)-1):
                    cv2.line(display_img, points[i], points[i+1], (255, 0, 0), 4)  # Blue color in BGR, thicker line
                # Add progress bar if tracker is provided
                if progress_tracker:
                    add_progress_bar_to_image(display_img, progress_tracker)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(points) > 2:
                # Create a temporary mask for the current polygon
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [np.array(points)], 255)
                # Add to the main mask
                mask = cv2.bitwise_or(mask, temp_mask)
                # Create blue overlay
                overlay = img.copy()
                overlay[temp_mask > 0] = [255, 0, 0]  # Blue color in BGR
                # Blend with original image
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, display_img)
                # Redraw instructions
                add_instructions(display_img)
                # Draw the outline
                cv2.polylines(display_img, [np.array(points)], True, (255, 0, 0), 4)  # Blue color in BGR, thicker line
                # Add progress bar if tracker is provided
                if progress_tracker:
                    add_progress_bar_to_image(display_img, progress_tracker)
    
    def add_instructions(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Add a semi-transparent background for better readability
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (600, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add instructions
        cv2.putText(img, "Instructions:", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "1. Click and drag to draw the wound outline.", (10, 60), font, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "2. Release when done.", (10, 90), font, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "3. Press ESC when finished.", (10, 120), font, 0.6, (255, 255, 255), 1)
    
    # Create window with a clean title
    cv2.namedWindow("Draw Mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Mask", draw_mask)
    
    # Add initial instructions
    add_instructions(display_img)
    
    # Initialize progress if tracker is provided
    if progress_tracker:
        progress_tracker.update_stage_progress("mask_drawing", 0.1)
        add_progress_bar_to_image(display_img, progress_tracker)
    
    while True:
        # Show the current image
        cv2.imshow("Draw Mask", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
    
    # Update progress when mask drawing is complete
    if progress_tracker:
        progress_tracker.update_stage_progress("mask_drawing", 1.0)
        # Set the chicken image and mask for the progress window
        progress_tracker.set_chicken_image(imgPath, mask)
    
    cv2.destroyAllWindows()
    return mask, img, mask
