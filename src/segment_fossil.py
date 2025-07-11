import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import csv
import cv2
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class CustomDataset(Dataset):
    """
    A custom dataset for extracting patches from a raster image (.tif).
    Only a central horizontal band is used, divided into tiles with an optional overlap.

    Attributes:
        input_raster (str): Path to input raster file (.tif).
        tile_size (int): Size (squared) of extracted tiles. (Default 640)
        stride (int): Horizontal offset between each tile. (Default 630)
        transforms (callable, optional): Transforms to be applied to each tile.
        raster_shape (tuple): Dimensions of original raster image.
        height (int): Image height (in px).
        width (int): Image width (in px).
        channels (int): Number of channels in image (default 3).
        num_tiles_x (int): Number of extracted horizontal tiles.
        num_tiles_y (int): Set to 1, as only a central strip is cut.
        center_y (int): Horizontal center position for extraction.
        tile_y_start (int): Horizontal start position for extracting center strip.
    """
    def __init__(self, input_raster, tile_size=640, stride=630, transforms=None):
        """
      Initializes dataset.

        Args:
            input_raster (str): Raster file path (.tif).
            tile_size (int): Tile size squared (default 640).
            stride (int): Horizontal offset between tiles. (Default 630).
            transforms (callable): Transforms to be applied to each tile (No transforms).
        """
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms

        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape  
        
        if len(self.raster_shape) == 2:
            self.height, self.width = self.raster_shape
            self.channels = 1
        
        else:
            self.height, self.width, self.channels = self.raster_shape

        self.num_tiles_x = max(1, (self.width - self.tile_size) // self.stride + 1)
        self.num_tiles_y = 1  
        self.center_y = self.height // 2  
        self.tile_y_start = max(0, self.center_y - self.tile_size // 2)  

    def __len__(self):
        """
        return the total number of horizontale tile 

        Returns:
            int: Total number of tile
        """
        return self.num_tiles_x  

    def __getitem__(self, idx):
        tile_x = self.width - self.tile_size - (idx * self.stride)
        tile_y = self.tile_y_start  
        tile_x = max(0, tile_x) 

        with tifffile.TiffFile(self.input_raster) as tif:
            band = tif.pages[0].asarray()
            # if len(band.shape) == 2:
                # band = band[..., np.newaxis]
                # band = np.repeat(band[:, :, np.newaxis], 3, axis=2)
            band = band[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        if self.transforms:
            band = self.transforms(band)
            
        # Convert image to RGB if it has 4 channels (RGBA)
        if band.shape[2] == 4:  # Sif rimage is RGBA
            band = band[:, :, :3]  # Only take the 3 first chanel

        return band

if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Traitement de segmentation des fossiles.")
#     parser.add_argument('--base_path', type=str, required=True, help='Chemin de base pour le traitement')
#     args = parser.parse_args()

#     base_path = args.base_path

    # Chemin du dossier contenant les images à traiter
    # base_path = "/home/killian/data2025/TGV4"
    # base_path = "/home/killian/data2025/TGV5"  
    # base_path = "/home/killian/data2025/15485"
    # base_path = "/home/killian/data2025/15492"  
    # base_path = "/home/killian/data2025/11478"  
    # base_path = "/home/killian/data2025/17689"  
    base_path = "/home/killian/data2025/Actual_Wood"
    
    # Get all files .tif
    image_paths = sorted(glob.glob(os.path.join(base_path, "*.tif")))  # Liste des fichiers TIF
    if not image_paths:
        print(f"Aucune image trouvée dans {base_path}")
        exit()

    # Extraire le nom du dossier parent
    base_folder = os.path.basename(base_path)  

    # Générer un dossier de sortie pour ce jeu de données
    output_dir = f"/home/killian/sam2/inferences/{base_folder}"
    os.makedirs(output_dir, exist_ok=True)       
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    # Créer un fichier CSV global pour chaques images
    csv_masks_file = os.path.join(output_dir, f"mask_measurements_{base_folder}.csv")
    with open(csv_masks_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    # Chargement du modèle
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    
    #Hyperparametres
    model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=40,  # Plus de points pour capturer les details
        points_per_batch=30,  # Augmenter pour calculer le nbre de points pris en meme temps (/!\ GPU)
        pred_iou_thresh=0.65,  # Reduire pour accepter plus de mask
        stability_score_thresh=0.80,  # Rzduire pour ne pas exclure trop de mask
        stability_score_offset=0.8,
        crop_n_layers=6,  # Ammeliore la segmentation des petites structures
        box_nms_thresh=0.60,  # Eviter la suppression excessive de petite structure
        crop_n_points_downscale_factor=1.5,  # Adapter aux images a haute resolution
        min_mask_region_area=6.0,  # Conserver plus de petits objets
        use_m2m=True,  # Mode avancé 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    size_threshold = 720 # Seuil pour filtrer les objets trop grands
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
            # Calcul de l'aire de chaque masque
            mask_areas = res_tensor.sum(dim=(1, 2)).cpu().numpy()

            # Calcul des bornes basées sur les percentiles
            lower_percentile = 5
            upper_percentile = 95
            min_area = np.percentile(mask_areas, lower_percentile)
            max_area = np.percentile(mask_areas, upper_percentile)

            # Filtrer les masques compris entre les bornes
            valid_mask_indices = (mask_areas >= min_area) & (mask_areas <= max_area)
            filtered_tensor = res_tensor[valid_mask_indices]

            masks_info = []

            # Extraction des objets de chaque masque
            for mask_id, mask_pred in enumerate(filtered_tensor):
                mask_np = mask_pred.numpy().astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                for j in range(1, num_labels):
                    x, y = centroids[j]
                    area = stats[j, cv2.CC_STAT_AREA]
                    equivalent_diameter = np.sqrt(4 * area / np.pi)

                    obj_mask = (labels == j)
                    masks_info.append((x, y, area, equivalent_diameter, mask_id, obj_mask))

            # Filtrage par distribution des aires
            areas = np.array([m[2] for m in masks_info])
            min_area = np.percentile(areas, 5)
            max_area = np.percentile(areas, 95)
            filtered_masks_info = [m for m in masks_info if min_area <= m[2] <= max_area]

            # Reconstruction d’un seul masque fusionné
            res_merge = np.zeros(res_tensor.shape[1:], dtype=bool)
            for _, _, _, _, _, obj_mask in filtered_masks_info:
                res_merge |= obj_mask

            # Tri des masques pour affichage ou traitement futur
            masks_info.sort(key=lambda m: (-m[1], -m[0]))

            csv_image_file = os.path.join(output_dir, f"mask_measurements_{image_name}.csv")
            file_exists = os.path.isfile(csv_image_file)

            # Sauvegarde du masque binaire dans le dossier mask/
            mask_output_path = os.path.join(mask_dir, f"{image_name}_Image_{i}_mask.tif")
            tifffile.imwrite(mask_output_path, res_merge.astype(np.uint8) * 255)

            # Sauvegarde des mesures
            with open(csv_image_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(["Tile_ID", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])
                
                unique_masks = {}
                for x, y, area, eq_diam, mask_id, _ in filtered_masks_info:
                    key = (round(x, 2), round(y, 2))
                    if key not in unique_masks or area > unique_masks[key][2]:
                        unique_masks[key] = (x, y, area, eq_diam)

                for new_id, (x, y, area, eq_diam) in enumerate(unique_masks.values()):
                    writer.writerow([f"Image_{i}", new_id, x, y, area, eq_diam])

            # Affichage des images avec matplotlib
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image_np)
            plt.title("Image Originale")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(res_merge.astype(np.uint8) * 255, cmap='gray')
            plt.title("Masques Prédits")
            plt.axis('off')

            output_img_path = os.path.join(output_dir, f"{image_name}_Image_{i}.png")
            plt.savefig(output_img_path, dpi=300)
            plt.close()

            # # Ajout des IDs des masques sur l'image des masques prédits
            # for x, y, area, equivalent_diameter, mask_id in masks_info:
            #     text = plt.text(
            #         x, y, str(mask_id), 
            #         color='blue', fontsize=6, fontweight='bold', ha='center', va='center'
            #     )
            #     # Ajout d'un contour noir pour améliorer la lisibilité
            #     text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

            # # Sauvegarde de l'image avec les IDs des masques annotés
            # output_img_path = os.path.join(output_dir, f"{image_name}_Image_{i}.png")
            # plt.savefig(output_img_path, dpi=300)  # Augmentation de la résolution pour plus de lisibilité
            # plt.close()
        
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

