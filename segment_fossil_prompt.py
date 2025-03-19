import os
import numpy as np
import tifffile

import csv
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils import get_centroids_threshold, plot_results


class CustomDataset(Dataset):
    def __init__(self, input_raster, tile_size=640, stride=10, transforms=None):
        
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms 
        
        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape  # (H, W, C)

        self.height, self.width, self.channels = self.raster_shape
        
        # Calculer le nombre de tuiles en hauteur et en largeur
        self.num_tiles_y = max(1, (self.height - self.tile_size) // self.stride + 1)
        self.num_tiles_x = max(1, (self.width - self.tile_size) // self.stride + 1)

    def __len__(self):
        return self.num_tiles_y * self.num_tiles_x

    def __getitem__(self, idx):
        # Convertir l'index linéaire en coordonnées 2D
        tile_y = (idx // self.num_tiles_x) * self.stride
        tile_x = (idx % self.num_tiles_x) * self.stride

        # S'assurer qu'on ne dépasse pas les bords de l'image
        tile_y = min(tile_y, self.height - self.tile_size)
        tile_x = min(tile_x, self.width - self.tile_size)

        with tifffile.TiffFile(self.input_raster) as tif:
            band = tif.pages[0].asarray()[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        if self.transforms:
            band = self.transforms(band)

        return band


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

if __name__ == "__main__":

    # checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_large.pt"
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    # checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # dataset paremeters
    path = '/home/killian/data2025/TGV4/X200_TGV4A_B_P.tif'
    tile_size = 640
    stride = 630

    dataset = CustomDataset(path, tile_size=tile_size, stride=stride)
    dataloader = DataLoader(dataset,batch_size=1, shuffle=False)

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    #Hyperparametres
    model = SAM2ImagePredictor(sam2,
                               mask_threshold=0,
                               max_hole_area=0,
                               max_sprinkle_area=0,
                               )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dossier de sortie
    output_dir = "/home/killian/sam2/inferences_prompt"
    os.makedirs(output_dir, exist_ok=True)

    # Fichiers CSV pour sauvegarder les masks
    csv_masks_file = os.path.join(output_dir, "mask_measurements.csv")

    with open(csv_masks_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_ID", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    size_threshold = 560  # Seuil pour filtrer les objets trop grands
    # (0.5060 px/µm) donc <280µm pour

    # Boucle d'inférence
    for i, image in enumerate(dataloader):

        if i == 34 :
            print(f"Image {i+1}/{len(dataloader)} - Avancement : {i/len(dataloader):.2%}")
            # print(f"Image {i+1}/{len(dataloader)} - Avancement : {i/len(dataloader):.2%}", end='\r')

            image_np = image.squeeze(0).cpu().numpy()
            centroids = get_centroids_threshold(image_np)
            print(centroids.shape)
            labels = np.ones(centroids.shape[0])
            
            model.set_image(image_np)
            masks, scores, logits = model.predict(
                point_coords=centroids,
                point_labels=labels,
                multimask_output=True,
            )

            
            filtered_tensor = masks[masks.sum(axis=(1, 2)) <= size_threshold]
            res_merge = filtered_tensor.any(axis=0)
            print(res_merge.shape)
            print(res_merge.sum())

            plot_results(os.path.join(output_dir, f"Image_{i}.png"), image_np, res_merge, centroids)
            
            # # Sauvegarde de l'image de prédiction et de l'image originale

            break
