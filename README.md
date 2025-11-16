# Suture-It

A 2D suture placement optimization tool for surgical wound closure planning.

## Quick Start

1. **Install dependencies**
   ```bash
   conda env create -f environment.yml
   conda activate <environment_name>
   ```
   Or manually:
   ```bash
   conda create --name suture-it python=3.10
   conda activate suture-it
   pip install Pillow opencv-python scikit-image matplotlib torch hydra-core tqdm torchvision tensorflow pandas trimesh iopath customtkinter
   ```

2. **Run the application**
   ```bash
   python main.py
   ```

## Usage

1. **Upload wound image** - Click "Upload Wound Image" and select an image file
2. **Trace wound outline** - Click and drag to draw the wound boundary (must be fully enclosed)
3. **Compute centerline** - Click "Begin Suture Planning" after tracing
4. **View prioritization points** (optional) - Review high-curvature areas and toggle whether to use them in optimization
5. **Run optimization** - Click "Begin Optimization" to generate suture plans
6. **View results** - Use the slider to compare different suture counts and their loss values

## Features

- Interactive wound tracing on uploaded images
- Automatic centerline detection
- Prioritization point identification for high-curvature areas
- Optimization algorithm that minimizes closure and shear forces
- Comparison of multiple suture plans with loss metrics
- Animated visualization showing prioritized sutures

## Interface Guide

- **Red points**: Insertion points
- **Blue points**: Extraction points  
- **Green lines**: Prioritized sutures (high-curvature areas)
- **Black lines**: Standard sutures
- **Slider**: Adjust number of sutures to compare plans
- **Loss display**: Shows normalized loss percentage for each plan

## Troubleshooting

- **App won't start**: Verify Python 3.10 and all dependencies are installed
- **Image upload fails**: Check file format (jpg, png, bmp, tiff, gif) and file integrity
- **Optimization errors**: Ensure wound outline is fully enclosed and clearly visible

## Important Notes

**Beta Software**: This is a beta test. Please send feedback to: **ria.jain@berkeley.edu**

**Current Limitations**:
- Does not consider skin deformations during implementation
- Does not account for patient movement
- Does not consider equipment material differences
- 2D only (no depth/3D consideration)

## Contact

Questions or feedback: **ria.jain@berkeley.edu**
