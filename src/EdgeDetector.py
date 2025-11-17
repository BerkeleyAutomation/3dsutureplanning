"""
EdgeDetector: Wound edge detection and centerline extraction.

This module processes wound images to extract the wound centerline using
skeletonization and spline fitting. Handles both 2D and 3D wound processing.
"""

from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize, medial_axis
import matplotlib.pyplot as plt
import scipy.interpolate as inter

from .point_ordering import get_pt_ordering
from .SAM import create_mask
from .largestCC import keep_largest_connected_component
from .fillHoles import fillHoles
from .utils import click_points_simple
import os


class EdgeDetector:
    """
    Detects wound edges and extracts centerline from images.
    
    Uses Canny edge detection, morphological operations, and skeletonization
    to find the wound centerline.
    """
    
    def find_edges(self, img):
        """
        Find edges in an image using Canny edge detection.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            np.array: Binary edge image
        """
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("grayscale_image.jpg", grayscale_image)
        cv2.imwrite("grayscale_image_clip.jpg", grayscale_image)
        cv2.imwrite("blur_clip.jpg", grayscale_image)
        return cv2.Canny(grayscale_image, 100, 600)

    def dilate_to_line(self, edge_mask, kernel_dim):
        """
        Dilate edge mask to create thicker lines.
        
        Args:
            edge_mask: Binary edge image
            kernel_dim: Size of dilation kernel
            
        Returns:
            np.array: Dilated edge image
        """
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
        return cv2.dilate(edge_mask, kernel, iterations=1)

    def generate_spline(self, pixels):
        """Placeholder for spline generation (not currently used)."""
        pass


def img_to_line(img_path, original_mask):
    """
    Extract wound centerline from image and mask.
    
    Processes the mask to find the skeleton (centerline), orders the points,
    and returns the centerline along with distance information for variable
    wound width calculations.
    
    Args:
        img_path (str): Path to wound image
        original_mask (np.array): Binary mask of wound region
        
    Returns:
        tuple: (ordered_points, mask_array, border_points, max_dist_hernia, ordered_points_dist)
            - ordered_points: Ordered centerline points
            - mask_array: Processed mask as numpy array
            - border_points: Wound border points with gaps filled
            - max_dist_hernia: Maximum distance from centerline to edge (for hernia detection)
            - ordered_points_dist: Points with distance to wound edge
    """
    # Save and process mask
    cv2.imwrite('data/temp_images/sam_mask.jpg', original_mask)
    
    # Keep only largest connected component
    mask = keep_largest_connected_component('data/temp_images/sam_mask.jpg')
    cv2.imwrite('data/temp_images/sam_mask.jpg', mask)
    
    # Extract border points for variable wound width calculations
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    border_pts = max(contours, key=cv2.contourArea).squeeze()

    # Post-process mask: dilate and fill holes
    new_edge_detector = EdgeDetector()
    mask = cv2.imread('data/temp_images/sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 5)
    cv2.imwrite("data/temp_images/dilated_sam.jpg", img_dilated)
    img_dilated = fillHoles('data/temp_images/dilated_sam.jpg')
    cv2.imwrite("data/temp_images/filledHoles.jpg", img_dilated)
    
    # Skeletonize to get centerline
    binary_image = np.where(img_dilated > 0, 1, 0)
    skeleton = skeletonize(binary_image)

    # Calculate distance transform for hernia detection
    # (distance from each point to nearest edge)
    skel, distance = medial_axis(binary_image, return_distance=True)
    dist_on_skel = distance * skel
    max_dist_hernia = np.max(dist_on_skel)

    # Save intermediate results
    np.save('data/temp_images/binary_skeleton.npy', skeleton)
    plt.imsave('data/temp_images/skeleton_sam.jpg', skeleton)

    # Order skeleton points along the centerline
    ordered_points, nonzero_pts = get_pt_ordering(skel)
    
    # Store points with their distance to wound edge
    ordered_points_dist = []
    for npt in nonzero_pts:
        ordered_points_dist.append((
            npt[0][1],  # y coordinate
            npt[0][0],  # x coordinate
            dist_on_skel[npt[0][1]][npt[0][0]]  # distance to edge
        ))

    filled_holes = Image.open("data/temp_images/sam_mask.jpg")
    numpydata = np.asarray(filled_holes)

    # Load and display image
    img = Image.open(img_path).resize((600, 400))
    left_img = np.asarray(img)
    plt.imshow(left_img)
    
    def fill_gaps(contour):
        """
        Fill gaps in contour by linear interpolation.
        
        If consecutive points are more than 2 pixels apart, interpolate
        intermediate points along the direction with more sparsity.
        
        Args:
            contour: Array of contour points
            
        Returns:
            np.array: Contour with gaps filled
        """
        def linear_int_x(x1, y1, x2, y2, y):
            """Linear interpolation for x given y."""
            return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

        def linear_int_y(x1, y1, x2, y2, x):
            """Linear interpolation for y given x."""
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

        def euc_dist(x, y):
            """Euclidean distance between two points."""
            return np.sqrt(abs(x[0] - y[0])**2 + abs(x[1] - y[1])**2)

        # Close the contour
        contour = np.append(contour, [contour[0]], axis=0)
        new_contour = np.copy(contour)
        
        # Fill gaps between consecutive points
        for i in range(len(contour) - 1):
            if euc_dist(contour[i], contour[i+1]) > 2:
                x1, x2, y1, y2 = contour[i][0], contour[i+1][0], contour[i][1], contour[i+1][1]
                # Interpolate along direction with more sparsity
                if abs(x1 - x2) > abs(y1 - y2):
                    # More horizontal - interpolate along x
                    for new_x in range(min(x1, x2) + 1, max(x1, x2)):
                        new_contour = np.append(
                            new_contour, 
                            [[new_x, int(linear_int_y(x1, y1, x2, y2, new_x))]], 
                            axis=0
                        )
                else:
                    # More vertical - interpolate along y
                    for new_y in range(min(y1, y2) + 1, max(y1, y2)):
                        new_contour = np.append(
                            new_contour, 
                            [[int(linear_int_x(x1, y1, x2, y2, new_y)), new_y]], 
                            axis=0
                        )
        return new_contour
       
    # Fill gaps in border points
    border_pts_gaps_filled = fill_gaps(border_pts)
    
    return ordered_points, numpydata, border_pts_gaps_filled, max_dist_hernia, ordered_points_dist


def line_to_spline(line, img_path, mm_per_pixel, viz=False):
    """
    Convert ordered centerline points to B-spline representation.
    
    Creates multiple spline representations:
    - Exact: Fits all points exactly
    - Sampled: Uses subset of points
    - Smoothed: Applies smoothing
    
    Args:
        line (list): Ordered centerline points
        img_path (str): Path to image (for visualization)
        mm_per_pixel (float): Conversion factor
        viz (bool): Whether to create visualization plots
        
    Returns:
        tuple: (sampled_spline_points, sampled_tck)
            - sampled_spline_points: Points along sampled spline
            - sampled_tck: Spline parameters
    """
    # Fit exact spline to all points
    exact_tck, u = inter.splprep(
        [[pt[0] for pt in line], [pt[1] for pt in line]], 
        k=3, 
        s=0
    )
    exact_wound_parametric = lambda t, d: inter.splev(t, exact_tck, der=d)

    # Sample points (1 in every 30)
    sample_ratio = 30
    sampled_pts = [line[i * sample_ratio] for i in range(len(line) // sample_ratio)] + [line[-1]]

    # Fit spline to sampled points
    sampled_tck, u = inter.splprep(
        [[pt[0] for pt in sampled_pts], [pt[1] for pt in sampled_pts]], 
        k=3, 
        s=0
    )
    sampled_wound_parametric = lambda t, d: inter.splev(t, sampled_tck, der=d)

    # Fit smoothed spline to all points
    smoothed_tck, u = inter.splprep(
        [[pt[0] for pt in line], [pt[1] for pt in line]], 
        k=3
    )
    smoothed_wound_parametric = lambda t, d: inter.splev(t, smoothed_tck, der=d)

    # Generate points along splines for visualization
    exact_spline_pts = []
    sampled_spline_pts = []
    smoothed_spline_pts = []

    for t_step in np.linspace(0, 1, 500):
        exact_spline_pts.append(exact_wound_parametric(t_step, 0))
        sampled_spline_pts.append(sampled_wound_parametric(t_step, 0))
        smoothed_spline_pts.append(smoothed_wound_parametric(t_step, 0))
    
    if viz:
        img = Image.open(img_path)
        img_np = np.asarray(img)
        plt.imshow(img_np)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img_np)
        ax1.plot(
            [pt[1]/mm_per_pixel for pt in exact_spline_pts], 
            [pt[0]/mm_per_pixel for pt in exact_spline_pts]
        )
        ax2.imshow(img_np)
        ax2.plot(
            [pt[1]/mm_per_pixel for pt in sampled_spline_pts], 
            [pt[0]/mm_per_pixel for pt in sampled_spline_pts]
        )
        ax3.imshow(img_np)
        ax3.plot(
            [pt[1]/mm_per_pixel for pt in smoothed_spline_pts], 
            [pt[0]/mm_per_pixel for pt in smoothed_spline_pts]
        )
        
        plt.savefig("spline.png")

    return sampled_spline_pts, sampled_tck


def line_to_spline_3d(line, sample_ratio=30, viz=False, s_factor=None):
    """
    Convert 3D centerline points to spline representation.
    
    Creates separate splines for x, y, and z coordinates parameterized
    by cumulative arc length.
    
    Args:
        line (np.array): 3D centerline points (N x 3)
        sample_ratio (int): Sampling ratio (currently unused)
        viz (bool): Whether to visualize (currently unused)
        s_factor (float): Smoothing factor for spline
        
    Returns:
        list: [x_spline, y_spline, z_spline] - UnivariateSpline objects
    """
    x = line[:, 0]
    y = line[:, 1]
    z = line[:, 2]

    # Calculate cumulative distance along curve
    distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

    # Normalize to 0-1 parameter range
    t = cumulative_distance / cumulative_distance[-1]

    # Fit separate splines for each dimension
    x_spline = inter.UnivariateSpline(t, x, s=s_factor)
    y_spline = inter.UnivariateSpline(t, y, s=s_factor)
    z_spline = inter.UnivariateSpline(t, z, s=s_factor)

    return [x_spline, y_spline, z_spline]
