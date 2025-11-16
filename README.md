# Suture-It

A 2D suture placement optimization algorithm and user interface for surgical wound closure planning.

## Overview

Suture-It is a medical software tool designed to help surgeons plan optimal suture placement for wound closure. The application uses computer vision and optimization algorithms to analyze wound images and suggest suture placement strategies that minimize closure forces and shear forces.

## Features

- **Interactive Wound Tracing**: Draw the outline of wounds directly on uploaded images
- **Automatic Centerline Detection**: Automatically computes the wound centerline for suture planning
- **Prioritization Points**: Identify and prioritize high-curvature areas of the wound
- **Optimization Algorithm**: Optimizes suture placement to minimize closure and shear forces
- **Multiple Suture Plans**: Compare different suture counts and their associated loss values
- **Animated Visualization**: Watch suture plans animate to see prioritized sutures highlighted
- **Loss Metrics**: View detailed loss calculations including total loss, closure loss, and shear loss

## Setup Instructions

### Prerequisites
- Python 3.10
- Conda (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BerkeleyAutomation/3dsutureplanning.git
   cd 3dsutureplanning
   ```

2. **Set up environment**

   **Option A: Using conda environment file (Recommended)**
   ```bash
   conda env create -f environment.yml
   conda activate <environment_name>
   ```

   **Option B: Manual setup**
   ```bash
   conda create --name suture-it python=3.10
   conda activate suture-it
   pip install Pillow opencv-python scikit-image matplotlib torch hydra-core tqdm torchvision tensorflow pandas trimesh iopath customtkinter
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## How to Use

### Step 1: Start the Application
Run `python main.py` to launch the Suture-It interface.

### Step 2: Upload a Wound Image
- Click "Upload Wound Image"
- Select an image file (supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`)
- The image will be displayed in the interface

### Step 3: Trace the Wound Outline
- Click and drag to trace the outline of the wound
- Release when done - the region should be fully enclosed in blue
- Click and drag again to retrace if needed

### Step 4: Compute Wound Centerline
- After tracing, the wound centerline will be displayed in red
- Click "Begin Suture Planning" to proceed

### Step 5: View Prioritization Points (Optional)
- The application will identify prioritization points (high-curvature areas)
- You can toggle whether to consider these points in optimization
- Click "View Prioritization Points" to see them highlighted
- Click "Begin Optimization" when ready

### Step 6: Optimization
- The optimization process will run, testing different numbers of sutures
- Progress is shown with a progress bar and detailed loss information
- You can watch the optimization in real-time

### Step 7: View Results
- Once optimization completes, click "View Optimized Suture Plan"
- Use the slider to compare different suture counts
- Each plan shows:
  - Number of sutures
  - Associated loss value
  - Visual representation with prioritized sutures in green
- The animation shows sutures appearing one by one, with prioritized sutures first

## Understanding the Interface

### Main Display Areas
- **Left Panel**: Optimization progress, loss metrics, and current suture plan visualization
- **Right Panel**: Final suture plan results with two views:
  - Uniform length sutures (simulated closed wound)
  - Variable length sutures (open wound with insert/extract points)

### Key Controls
- **Slider**: Adjust the number of sutures to compare different plans
- **Toggle Switch**: Enable/disable consideration of prioritization points in optimization
- **Loss Display**: Shows normalized loss percentages for each suture plan

### Color Coding
- **Red points**: Insertion points
- **Blue points**: Extraction points
- **Green lines**: Prioritized sutures (high-curvature areas)
- **Black lines**: Standard sutures

## File Structure

- `main.py`: Main GUI application
- `SuturePlacer.py`: Core optimization algorithm
- `EdgeDetector.py`: Wound edge detection and centerline computation
- `Constraints.py`: Suture placement constraints
- `RewardFunction.py`: Loss function calculations
- `temp_images/`: Temporary files generated during processing

## Troubleshooting

### Application won't start
- Ensure all dependencies are installed correctly
- Check that Python 3.10 is being used
- Verify that `temp_images/` directory exists (it should be created automatically)

### Image upload issues
- Ensure image file is in a supported format
- Check that the image file is not corrupted
- Try a different image format if issues persist

### Optimization errors
- Make sure the wound outline is fully enclosed
- Ensure the wound region is clearly visible in the image
- Try retracing the wound outline if optimization fails

## Important Notes

### Beta Software
This is a beta test of software in progress. Please send feedback to: **ria.jain@berkeley.edu**

### Limitations
In its current stage, Suture-It does not consider:
- Potential skin deformations during implementation
- Dynamic skin surface and patient movement during procedure
- Differences in equipment materials (needle, thread, etc.)
- Depth/3D conceptualization of wound

### Future Work
Potential enhancements include:
- Dynamic adjustments to suture plan during suturing
- Projection of suture plan onto wound for real-time guidance

## Contact

For questions, feedback, or issues, please contact: **ria.jain@berkeley.edu**

## License

[Add license information here]
