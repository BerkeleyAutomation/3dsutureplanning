# Suture-Placement
2D Suture placement optimization algorithm and user interface.

## Setup Instructions
1. Clone repo
   ```
   git clone https://github.com/BerkeleyAutomation/3dsutureplanning.git
   ```
2. Set up environment
   * Create and activate a new conda environment with Python 3.10 & required packages
     ```
     conda create --name <environment_name> python=3.10
     conda activate <environment_name>
     pip install Pillow opencv-python scikit-image matplotlib torch hydra-core tqdm torchvision tensorflow pandas trimesh iopath customtkinter
     ```
   OR
   * run ```conda env create -f environment.yml``` to use the repository environment
3. Open repository directory and run program
   ```
   cd 3dsutureplanning
   python main.py
   ```
