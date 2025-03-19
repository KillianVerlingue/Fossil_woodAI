import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import csv
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class CustomDataset(Dataset):
    def __init__(self, input_raster, tile_size=640, stride=10, transforms=None):
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms 

        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape  # (H, W, C)

        self.height, self.width, self.channels = self.raster_shape

        # Ne garder qu'une seule colonne de tuiles centrée
        self.num_tiles_y = max(1, (self.height - self.tile_size) // self.stride + 1)
        self.num_tiles_x = 1  # On ne prend qu'une seule bande verticale
        self.center_x = self.width // 2  # Centre de l'image
        self.tile_x_start = max(0, self.center_x - self.tile_size // 2)  # Assurer qu'on ne sort pas des bords

    def __len__(self):
        return self.num_tiles_y  # On ne parcourt que les tuiles en hauteur

    def __getitem__(self, idx):
        tile_y = idx * self.stride
        tile_x = self.tile_x_start  # Toujours la même valeur

        # S'assurer qu'on ne dépasse pas les bords
        tile_y = min(tile_y, self.height - self.tile_size)

        with tifffile.TiffFile(self.input_raster) as tif:
            band = tif.pages[0].asarray()[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        if self.transforms:
            band = self.transforms(band)

        return band


if __name__ == "__main__":

    # checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_large.pt"
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    # checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # dataset paremeters
    path = '/home/killian/data2025/15485/X200_15485_PB1.tif'
    tile_size = 640
    stride = 630

    dataset = CustomDataset(path, tile_size=tile_size, stride=stride)
    dataloader = DataLoader(dataset,batch_size=1, shuffle=False)

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    #Hyperparametres
    model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=30,  # Plus de points pour capturer les details
        points_per_batch=20,  # Augmenter pour calculer le nbre de points pris en meme temps (/!\ GPU)
        pred_iou_thresh=0.5,  # Reduire pour accepter plus de mask
        stability_score_thresh=0.70,  # Rzduire pour ne pas exclure trop de mask
        stability_score_offset=0.8,
        crop_n_layers=4,  # Ammeliore la segmentation des petites structures
        box_nms_thresh=0.60,  # Eviter la suppression excessive de petite structure
        crop_n_points_downscale_factor=1.5,  # Adapter aux images a haute resolution
        min_mask_region_area=10.0,  # Conserver plus de petits objets
        use_m2m=True,  # Mode avancé 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dossier de sortie
    output_dir = "/home/killian/sam2/inferences"
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
        print(f"Image {i+1}/{len(dataloader)} - Avancement : {i/len(dataloader):.2%}")
        # print(f"Image {i+1}/{len(dataloader)} - Avancement : {i/len(dataloader):.2%}", end='\r')

        image_np = image.squeeze(0).cpu().numpy()
        pred = model.generate(image_np)
        res_tensor = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
        filtered_tensor = res_tensor[res_tensor.sum(dim=(1, 2)) <= size_threshold]
        res_merge = filtered_tensor.any(dim=0)
        
        # Trier les masques de droite à gauche selon la coordonnée X de leur centroïde
        masks_info = []
        for mask_id, mask_pred in enumerate(filtered_tensor):
            mask_np = mask_pred.numpy().astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            
            for j in range(1, num_labels):  # On ignore le label 0 (fond)
                x, y = centroids[j]
                area = stats[j, cv2.CC_STAT_AREA]
                equivalent_diameter = np.sqrt(4 * area / np.pi)
                masks_info.append((x, y, area, equivalent_diameter, mask_id))
                
        # Trier les masques en fonction de X (croissant)
        masks_info.sort(reverse=True, key=lambda m: m[0])
        
        # Sauvegarde des masques triés
        with open(csv_masks_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for x, y, area, equivalent_diameter, mask_id in masks_info:
                writer.writerow([f"Image_{i}", mask_id, x, y, area, equivalent_diameter])
        
        # Sauvegarde de l'image de prédiction et de l'image originale
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title("Image Originale")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(res_merge.cpu().numpy(), cmap='gray')
        plt.title("Masques Prédits")
        plt.axis('off')
        
        plt.savefig(os.path.join(output_dir, f"Image_{i}.png"))
        plt.close()
