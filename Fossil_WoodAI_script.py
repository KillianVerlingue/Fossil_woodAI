import os
import glob
import csv
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Configuration des paramètres globaux
points_per_side = 32
size_threshold = 50000
distance_threshold = 2.0
tolerance = 10
N = 50
distance_threshold_line = 15

# Configuration des hyperparamètres pour SAM2
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

# Classe CustomDataset pour charger les images
class CustomDataset(Dataset):
    def __init__(self, input_raster, tile_size=640, stride=630, transforms=None):
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
        if band.shape[2] == 4:
            band = band[:, :, :3]
        return band

# Initialisation des répertoires de base
base_paths = ['/home/killian/data2025/11478']
output_base = '/home/killian/sam2/inferences'
os.makedirs(output_base, exist_ok=True)

# Workflow de traitement
for base_path in base_paths:
    fossil_name = os.path.basename(base_path)
    output_dir = os.path.join(output_base, fossil_name)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(base_path, '*.tif')))

    # Liste pour collecter les données de toutes les tuiles d'une image
    all_data_per_image = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path).split('.')[0]
        dataset = CustomDataset(image_path)
        dataloader = DataLoader(dataset, batch_size=1)

        for i, tile in enumerate(dataloader):
            tile_np = tile.squeeze(0).cpu().numpy()
            model = SAM2AutomaticMaskGenerator(sam2, points_per_side=points_per_side)
            pred = model.generate(tile_np)
            filtered_masks = [m for m in pred if np.sum(m['segmentation']) <= size_threshold]

            # Liste pour stocker les centroïdes et les contours de toutes les cellules détectées
            centroids = []
            contours_list = []

            for mask_id, mask in enumerate(filtered_masks):
                mask_np = mask['segmentation'].astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
                contours_list.append(cnt)

            # Création de la figure pour afficher les images et les masques
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Affichage de l'image originale
            ax[0].imshow(tile_np)
            ax[0].set_title('Image originale (tuile)')
            ax[0].axis('off')

            # Affichage du masque
            mask_overlay = np.zeros_like(tile_np)
            for mask in filtered_masks:
                mask_np = mask['segmentation'].astype(np.uint8)
                mask_overlay = cv2.bitwise_or(mask_overlay, mask_np)
            ax[1].imshow(mask_overlay, cmap='jet', alpha=0.5)
            ax[1].set_title('Masque')
            ax[1].axis('off')

            # Marquer les centroïdes des cellules choisies en rouge
            for centroid in centroids:
                ax[1].plot(centroid[0], centroid[1], 'ro')

            # Annoter les numéros des masques
            for mask_id, centroid in enumerate(centroids):
                ax[1].text(centroid[0] + 5, centroid[1] + 5, str(mask_id), color='white', fontsize=10)

            # Marquer la ligne cellulaire avec les cellules choisies en rouge
            for j in range(len(centroids) - 1):
                ax[1].plot([centroids[j][0], centroids[j+1][0]], [centroids[j][1], centroids[j+1][1]], 'r-', lw=2)

            # Affichage des images
            plt.tight_layout()
            plt.show()

            # Calcul des distances entre les centroïdes des cellules voisines
            for j in range(len(centroids) - 1):
                cx1, cy1 = centroids[j]
                cx2, cy2 = centroids[j + 1]
                
                # Calcul de la distance brute entre les centroïdes
                distance = np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2]))

                # Récupérer les contours des deux cellules pour estimer la distance utile
                mask1 = contours_list[j]
                mask2 = contours_list[j + 1]

                # Estimer la distance utile entre les deux centroïdes (ajustée par la portion dans le masque)
                distance_utile = distance - (cv2.contourArea(mask1) ** 0.5 + cv2.contourArea(mask2) ** 0.5)

                # Limiter la distance utile à 0 si elle devient négative
                distance_utile = max(0, distance_utile)

                # Ajout des mesures dans la liste des données de cette image
                area1 = cv2.contourArea(mask1)
                equivalent_diameter1 = np.sqrt(4 * area1 / np.pi)
                area2 = cv2.contourArea(mask2)
                equivalent_diameter2 = np.sqrt(4 * area2 / np.pi)

                all_data_per_image.append([f"Tile_{i}", f"Mask_{j}", centroids[j][0], centroids[j][1], distance_utile, area1, equivalent_diameter1])
                all_data_per_image.append([f"Tile_{i}", f"Mask_{j+1}", centroids[j+1][0], centroids[j+1][1], distance_utile, area2, equivalent_diameter2])

        # Une fois toutes les tuiles traitées, concaténer et enregistrer
        if all_data_per_image:
            csv_file = os.path.join(output_dir, f'{image_name}_cellular_line_data.csv')
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Tile_ID', 'Mask_ID', 'Centroid_X', 'Centroid_Y', 'Double_Paroi_Distance', 'Area', 'Equivalent_Diameter'])
                writer.writerows(all_data_per_image)

            print(f'Traitement de {image_name} terminé, données sauvegardées dans {csv_file}')
        
        # Reset pour la prochaine image
        all_data_per_image = []
