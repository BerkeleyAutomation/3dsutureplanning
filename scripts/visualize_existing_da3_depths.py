"""
Visualize existing DA3 depth maps from dan/dan_depth directory.

This creates PNG visualization images from the existing .npy depth files.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import glob

project_dir = "/Users/preethi/Documents/research/3dsutureplanning"
sys.path.insert(0, project_dir)

output_dir = os.path.join(project_dir, "data/da3_results")
os.makedirs(output_dir, exist_ok=True)


def generate_depth_visualization(depth_map, save_path):
    """Create a colorized visualization of the depth map."""
    # Normalize depth for visualization
    depth_vis = np.zeros_like(depth_map)
    
    # Handle negative values (some depth maps might be inverted)
    if depth_map.min() < 0:
        # If negative, invert and normalize
        depth_map = -depth_map
    
    if (depth_map > 0).any():
        d_min, d_max = depth_map[depth_map > 0].min(), depth_map[depth_map > 0].max()
        if d_max > d_min:
            depth_vis = (depth_map - d_min) / (d_max - d_min)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Save
    Image.fromarray(depth_colored).save(save_path)
    return depth_colored


def main():
    """Visualize existing DA3 depth maps."""
    print("=" * 70)
    print("Visualizing Existing DA3 Depth Maps")
    print("=" * 70)
    
    # Find depth files
    depth_dir = os.path.join(project_dir, "dan/dan_depth")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_image_*.npy")))
    
    if not depth_files:
        print(f"✗ No depth files found in {depth_dir}")
        return
    
    print(f"\nFound {len(depth_files)} depth files")
    print(f"Output directory: {output_dir}\n")
    
    # Process up to 3 files (as user requested)
    files_to_process = depth_files[:3]
    
    for i, depth_file in enumerate(files_to_process, 1):
        base_name = os.path.splitext(os.path.basename(depth_file))[0]
        print(f"[{i}/{len(files_to_process)}] Processing: {base_name}")
        
        # Load depth map
        depth = np.load(depth_file)
        print(f"  ✓ Loaded: shape {depth.shape}, range [{depth.min():.4f}, {depth.max():.4f}]")
        
        # Copy .npy file to output directory
        output_npy = os.path.join(output_dir, f"{base_name}.npy")
        np.save(output_npy, depth)
        print(f"  ✓ Saved: {output_npy}")
        
        # Create visualization
        output_png = os.path.join(output_dir, f"{base_name}.png")
        generate_depth_visualization(depth, output_png)
        print(f"  ✓ Saved visualization: {output_png}")
    
    print("\n" + "=" * 70)
    print(f"✓ Created visualizations for {len(files_to_process)} depth maps")
    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

