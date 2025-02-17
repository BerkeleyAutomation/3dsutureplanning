
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

def create_mask(imgPath, points, labels, ax, box_method=False): #point is the point where we floodfill from.
    def show_mask(mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)
        plt.show()
        
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

    device = "cpu"

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam.to(device=device)
    predictor1 = SAM2ImagePredictor(sam)
    img1_path = imgPath
    img1 = cv2.imread(img1_path)

    predictor1.set_image(img1)
    input_point = points
    input_label = labels
    masks_left, scores_left, logits_left = predictor1.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    img_trans = np.transpose(masks_left, (1, 2, 0))
    masks_left = masks_left[0]
    masked_left_img = img1*img_trans
    zero_one_mask = np.ones_like(masks_left, dtype=int) * masks_left * 256
    
    return zero_one_mask, cv2.cvtColor(masked_left_img, cv2.COLOR_BGR2RGB), masks_left
