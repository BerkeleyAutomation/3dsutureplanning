# Unused Files - Safe to Delete

## Files NOT Used by main.py (Main Application)

### Standalone Scripts/Experiments (Not imported anywhere)
1. `SutureDisplayAdjust.py` - Not imported in main.py (commented out)
2. `test_force_model.py` - Test/experiment file
3. `viz.py` - Standalone visualization script
4. `createPoints.py` - Standalone point picker utility
5. `convert_to_xyz.py` - Standalone conversion script
6. `image_transform.py` - Standalone transform utility
7. `force_model_new.py` - Standalone force model
8. `enhance_image.py` - Standalone image enhancement
9. `intersect.py` - Standalone intersection utility
10. `MeshIngestor.py` - Only used by test_force_model.py
11. `Kimhs_run.py` - Standalone run script
12. `SAMTest.py` - Test file for SAM

### 3D Pipeline (Separate workflow, not used by main.py)
13. `3D/3dpipeline.py` - Standalone 3D pipeline script
14. `3D/Optimizer3d.py` - Only used by 3dpipeline.py
15. `3D/RewardFunction3D.py` - Only used by 3dpipeline.py
16. `3D/SuturePlacement3d.py` - Only used by 3dpipeline.py

### Archive Directory (Old/Deprecated Files)
17. `Archive/` - Entire directory contains old versions and experiments
   - All files in Archive/ are safe to delete

### CGAL Directory (C++ Library)
18. `CGAL-5.6/` - C++ computational geometry library, not used by Python app

### Data/Output Files (Generated, can be regenerated)
19. `clicked_losses.csv` - Generated output file
20. `clicked_points.txt` - Generated output file
21. `binary_skeleton.npy` - Generated file
22. `surrounding_pts.npy` - Generated file
23. `loss.json` - Generated file
24. `adjacency_matrix.txt` - Generated file
25. `synth_adjacency.txt` - Generated file
26. `synth_coordinates.txt` - Generated file
27. `vertex_lookup.txt` - Generated file
28. `pipeline_xyz_pts.xyz` - Generated file
29. `xyz_pts.xyz` - Generated file (if exists)

### Build Files
30. `build/` - CMake build directory (can be regenerated)
31. `CMakeLists.txt` - Only needed for C++ builds
32. `generate_mesh` - Compiled binary
33. `generate_mesh.cpp` - C++ source (if not needed)

### Image Files (Check if needed for UI)
- `longS.png` - Check if used (might be example)
- `shortS_far.png` - Check if used
- `Vshape_far.png` - Check if used
- `chicken_images/` - Example/test data (might want to keep some)
- `chicken_skin_8-21-25/` - Example data
- `dan/` - Example data directory
- `images/` - Check if contains examples or actual data
- `real_images/` - Example data
- `extra_images/` - Example data
- `roast_chicken.glb` - 3D model file (if not used)

### Other Directories
- `insertion_extraction_pts/` - Generated data
- `point_cloud_data/` - Generated data
- `transforms/` - Generated data
- `masks/` - Generated data (if exists)

## Files KEPT (Used by main.py)

### Core Application Files
- `main.py` - Main entry point
- `SuturePlacer.py` - Core optimization
- `EdgeDetector.py` - Wound detection
- `SAM.py` - Mask creation
- `DistanceCalculator.py` - Distance calculations
- `RewardFunction.py` - Loss function
- `Constraints.py` - Suture constraints
- `point_ordering.py` - Point ordering
- `largestCC.py` - Connected component
- `fillHoles.py` - Image processing
- `utils.py` - Utility functions

### Required Directories
- `sam2/` - SAM2 model (required by SAM.py)
- `temp_images/` - Runtime directory (auto-created)
- `plots/` - Output directory

### UI Assets (Required)
- `logo2.png` - Logo image
- `external_link.png` - Icon
- `resulting_sutures.png` - Example image
- `example_uploads.png` - Example image

### Config Files
- `environment.yml` - Conda environment
- `requirements.txt` - Python dependencies
- `README.md` - Documentation

