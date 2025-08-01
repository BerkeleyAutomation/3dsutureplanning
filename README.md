# Suture-Placement

## Setup Instructions
1. Clone repo
   ```
   git clone https://github.com/BerkeleyAutomation/3dsutureplanning.git
   ```
3. Create and activate a new conda environment with Python 3.10 & required packages
   ```
   conda create --name <environment_name> python=3.10
   conda activate <environment_name>
   pip install Pillow opencv-python scikit-image matplotlib torch hydra-core tqdm torchvision tensorflow pandas trimesh iopath customtkinter
   ```
   OR
   run ```conda env create -f environment.yml``` to use the repository environment
4. Open repository directory and run program
   ```
   cd 3dsutureplanning
   python main.py
   ```
