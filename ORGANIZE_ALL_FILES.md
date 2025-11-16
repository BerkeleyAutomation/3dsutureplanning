# File Organization System

```
3dsutureplanning/
├── main.py                    # Main application (keep in root)
├── README.md                  # Documentation (keep in root)
├── requirements.txt           # Dependencies (keep in root)
├── environment.yml            # Conda env (keep in root)
│
├── src/                       # Core application code
│   ├── SuturePlacer.py
│   ├── EdgeDetector.py
│   ├── SAM.py
│   ├── DistanceCalculator.py
│   ├── RewardFunction.py
│   ├── Constraints.py
│   ├── point_ordering.py
│   ├── largestCC.py
│   ├── fillHoles.py
│   └── utils.py
│
├── assets/                    # All assets
│   ├── ui/                    # UI images (already organized)
│   ├── examples/              # Example images (already organized)
│   ├── generated/             # Generated outputs (already organized)
│   ├── generated_outputs/     # Old generated (already organized)
│   └── test_data/             # Test data (already organized)
│
├── data/                      # Data files
│   ├── insertion_extraction_pts/
│   ├── point_cloud_data/
│   ├── transforms/
│   └── temp_images/           # Runtime temp files
│
├── config/                    # Configuration files
│   ├── synth_adjacency.txt
│   ├── synth_coordinates.txt
│   ├── adjacency_matrix.txt
│   └── vertex_lookup.txt
│
├── scripts/                    # Standalone scripts
│   ├── enhance_image.py
│   ├── createPoints.py
│   ├── convert_to_xyz.py
│   ├── image_transform.py
│   ├── intersect.py
│   └── MeshIngestor.py
│
├── experiments/               # Experimental/test code
│   ├── test_force_model.py
│   ├── SAMTest.py
│   ├── Kimhs_run.py
│   └── viz.py
│
├── 3D/                        # 3D pipeline (keep as-is)
├── Archive/                   # Old code (keep as-is)
├── sam2/                      # SAM2 library (keep as-is)
└── CGAL-5.6/                  # CGAL library (keep as-is)
```

