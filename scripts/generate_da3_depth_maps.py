"""
Generate DA3 depth maps for chicken images.

This script processes chicken images and generates depth maps using DA3,
saving both .npy files and visualization images.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

project_dir = "/Users/preethi/Documents/research/3dsutureplanning"
sys.path.insert(0, project_dir)

# Add Depth-Anything-3 to path if it exists
da3_src_dir = os.path.join(project_dir, "Depth-Anything-3", "src")
if os.path.exists(da3_src_dir):
    sys.path.insert(0, da3_src_dir)

output_dir = os.path.join(project_dir, "data/da3_results")
os.makedirs(output_dir, exist_ok=True)


def generate_depth_visualization(depth_map, save_path):
    """Create a colorized visualization of the depth map."""
    # Normalize depth for visualization
    depth_vis = np.zeros_like(depth_map)
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


def process_image_with_da3(image_path, output_dir):
    """Process a single image with DA3 and save depth map."""
    try:
        import torch
        # Import directly from api module
        from depth_anything_3.api import DepthAnything3
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"  Using device: {device}")
        
        # Load model
        print("  Loading DA3 model...")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device)
        model.eval()
        
        # Run inference
        print(f"  Processing {os.path.basename(image_path)}...")
        prediction = model.inference([image_path])
        
        # Extract depth map
        depth = prediction.depth[0]  # Get first (and only) depth map
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save depth map as .npy
        depth_npy_path = os.path.join(output_dir, f"{base_name}_depth.npy")
        np.save(depth_npy_path, depth)
        print(f"  ✓ Saved depth map: {depth_npy_path}")
        
        # Save visualization
        depth_png_path = os.path.join(output_dir, f"{base_name}_depth.png")
        generate_depth_visualization(depth, depth_png_path)
        print(f"  ✓ Saved visualization: {depth_png_path}")
        
        print(f"  ✓ Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        
        return depth
        
    except Exception as e:
        print(f"  ✗ Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Generate DA3 depth maps for chicken images."""
    print("=" * 70)
    print("Generating DA3 Depth Maps for Chicken Images")
    print("=" * 70)
    
    # Find chicken images
    chicken_dir = os.path.join(project_dir, "chicken_images")
    if not os.path.exists(chicken_dir):
        chicken_dir = os.path.join(project_dir, "dan/dan_chicken")
    
    if not os.path.exists(chicken_dir):
        print(f"✗ Chicken images directory not found")
        return
    
    # Find left images (we'll process left images for depth)
    import glob
    left_images = sorted(glob.glob(os.path.join(chicken_dir, "left_exp_*.png")))
    
    if not left_images:
        print(f"✗ No left_exp_*.png images found in {chicken_dir}")
        return
    
    print(f"\nFound {len(left_images)} chicken images to process")
    print(f"Output directory: {output_dir}\n")
    
    # Process up to 3 images (as user requested earlier)
    images_to_process = left_images[:3]
    
    for i, image_path in enumerate(images_to_process, 1):
        print(f"\n[{i}/{len(images_to_process)}] Processing: {os.path.basename(image_path)}")
        process_image_with_da3(image_path, output_dir)
    
    print("\n" + "=" * 70)
    print(f"✓ Generated DA3 depth maps for {len(images_to_process)} images")
    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

