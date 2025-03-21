import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import csv
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class CustomDataset(Dataset):
    def __init__(self, input_raster, tile_size=640, stride=10, transforms=None):
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms 

        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape  

        self.height, self.width, self.channels = self.raster_shape

        self.num_tiles_x = max(1, (self.width - self.tile_size) // self.stride + 1)
        self.num_tiles_y = 1  
        self.center_y = self.height // 2  
        self.tile_y_start = max(0, self.center_y - self.tile_size // 2)  

    def __len__(self):
        return self.num_tiles_x  

    def __getitem__(self, idx):
        tile_x = self.width - self.tile_size - (idx * self.stride)
        tile_y = self.tile_y_start  
        tile_x = max(0, tile_x) 

        with tifffile.TiffFile(self.input_raster) as tif:
            band = tif.pages[0].asarray()[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        if self.transforms:
            band = self.transforms(band)

        return band

if __name__ == "__main__":
    
    # Chemin du dossier contenant les images à traiter
    base_path = "/home/killian/data2025/TGV4"  # Modifier selon ton besoin

    # Récupérer tous les fichiers .tif
    image_paths = sorted(glob.glob(os.path.join(base_path, "*.tif")))  # Liste des fichiers TIF
    if not image_paths:
        print(f"Aucune image trouvée dans {base_path}")
        exit()

    # Extraire le nom du dossier parent
    base_folder = os.path.basename(base_path)  

    # Générer un dossier de sortie pour ce jeu de données
    output_dir = f"/home/killian/sam2/inferences/{base_folder}"
    os.makedirs(output_dir, exist_ok=True)

    # Créer un fichier CSV global pour chaques images
    csv_masks_file = os.path.join(output_dir, f"mask_measurements_{base_folder}.csv")
    with open(csv_masks_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    # Chargement du modèle
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    
    #Hyperparametres
    model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=30,  # Plus de points pour capturer les details
        points_per_batch=30,  # Augmenter pour calculer le nbre de points pris en meme temps (/!\ GPU)
        pred_iou_thresh=0.6,  # Reduire pour accepter plus de mask
        stability_score_thresh=0.80,  # Rzduire pour ne pas exclure trop de mask
        stability_score_offset=0.8,
        crop_n_layers=4,  # Ammeliore la segmentation des petites structures
        box_nms_thresh=0.70,  # Eviter la suppression excessive de petite structure
        crop_n_points_downscale_factor=1.5,  # Adapter aux images a haute resolution
        min_mask_region_area=15.0,  # Conserver plus de petits objets
        use_m2m=True,  # Mode avancé 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    size_threshold = 560  # Seuil pour filtrer les objets trop grands
    # (0.5060 px/µm) donc <280µm pour

    # Boucle sur toutes les images du dossier
    for path in image_paths:
        image_name = os.path.basename(path)  
        print(f"Traitement de {image_name}...")

        dataset = CustomDataset(path, tile_size=640, stride=630)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, image in enumerate(dataloader):
            print(f"  Image {i+1}/{len(dataloader)} - Avancement : {i/len(dataloader):.2%}")

            image_np = image.squeeze(0).cpu().numpy()
            pred = model.generate(image_np)
            res_tensor = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
            filtered_tensor = res_tensor[res_tensor.sum(dim=(1, 2)) <= size_threshold]
            res_merge = filtered_tensor.any(dim=0)

            masks_info = []
            for mask_id, mask_pred in enumerate(filtered_tensor):
                mask_np = mask_pred.numpy().astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
                
                for j in range(1, num_labels):
                    x, y = centroids[j]
                    area = stats[j, cv2.CC_STAT_AREA]
                    equivalent_diameter = np.sqrt(4 * area / np.pi)
                    masks_info.append((x, y, area, equivalent_diameter, mask_id))
                    
            masks_info.sort(reverse=True, key=lambda m: m[0])

            with open(csv_masks_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, area, equivalent_diameter, mask_id in masks_info:
                    writer.writerow([image_name, mask_id, x, y, area, equivalent_diameter])

            # Sauvegarde des images
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image_np)
            plt.title("Image Originale")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(res_merge.cpu().numpy(), cmap='gray')
            plt.title("Masques Prédits")
            plt.axis('off')

            output_img_path = os.path.join(output_dir, f"{image_name}_Image_{i}.png")
            plt.savefig(output_img_path)
            plt.close()

    print(f"Traitement du dossier {output_dir} terminé ")
