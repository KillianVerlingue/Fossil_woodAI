import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_large.pt"
checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_tiny.pt"
# checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # predictor.set_image("/home/killian/sam2")
    # masks, _, _ = predictor.predict("0")
    # 🔹 Chemins vers le modèle et la configuration
#checkpoint = "./checkpoints/sam2_hiera_large.pt"
#model_cfg = "sam2_hiera_l.yaml"

import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import tifffile

class CustomDataset(Dataset):

    def __init__(self, input_raster, tile_size=640, stride=10, transforms=None):
        # Récupère les informations sur l'image
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms

        # Charger seulement les dimensions de l'image pour éviter de la charger entièrement
        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape
        print(self.raster_shape)

    def __len__(self):
        # Calculer le nombre de tuiles que l'on peut extraire de l'image
        return (self.raster_shape[1] - self.tile_size) // self.stride + 1

    def __getitem__(self, idx):
        # Définir la largeur de la bande centrale à charger 
        center_width = 640
        center_height = self.tile_size  # 640 de haut

        # Calculer les indices pour la bande centrale 
        start_x = max(0, (self.raster_shape[1] // 2) - (center_width // 2))
        end_x = start_x + center_width
        start_y = (self.raster_shape[0] // 2) - (center_height // 2)
        end_y = start_y + center_height

        # Lire seulement la bande centrale de l'image
        with tifffile.TiffFile(self.input_raster) as tif:
            # Charger cette portion spécifique de l'image (bande centrale)
            band = tif.pages[0].asarray()[start_y:end_y, start_x:end_x, :]

        # Découper la bande pour obtenir une tuile de taille (640x640)
        data = band[:, :, :]  # Ici, tu prends toute la largeur de la bande et la hauteur complète

        # Appliquer la transformation si nécessaire
        data = np.transpose(data, (1, 2, 0))  # Passer de (C, H, W) à (H, W, C)

        if self.transforms:
            data = self.transforms(data)

        return data


from torch.utils.data import DataLoader

# dataset = CustomDataset('/home/killian/data2025/15485/X200_15485_PB1.tif', tile_size=640, stride=10)
dataset = CustomDataset('/home/killian/data2025/TGV4/X200_TGV4B_B-P_2.tif', tile_size=640, stride=10)
dataloader = DataLoader(dataset,batch_size=1, shuffle=False)

#Hyperparametres
model = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=10,  # Plus de points pour capturer les details
    points_per_batch=10,  # Augmenter pour calculer le nbre de points pris en meme temps (/!\ GPU)
    pred_iou_thresh=0.7,  # Reduire pour accepter plus de mask
    stability_score_thresh=0.70,  # Rzduire pour ne pas exclure trop de mask
    stability_score_offset=0.8,
    crop_n_layers=2,  # Ammeliore la segmentation des petites structures
    box_nms_thresh=0.70,  # Eviter la suppression excessive de petite structure
    crop_n_points_downscale_factor=1,  # Adapter aux images a haute resolution
    min_mask_region_area=15.0,  # Conserver plus de petits objets
    use_m2m=True,  # Mode avancé 
)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

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
    image_np = image_np.swapaxes(2, 1)
    print(image_np.shape)
    pred = model.generate(image_np)
    res_tensor = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
    filtered_tensor = res_tensor[res_tensor.sum(dim=(1, 2)) <= size_threshold]
    res_merge = filtered_tensor.any(dim=0).to(device)
    
    # Trier les masques de droite à gauche selon la coordonnée X de leur centroïde
    masks_info = []
    for mask_id, mask_pred in enumerate(filtered_tensor):
        mask_np = mask_pred.cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        
        for j in range(1, num_labels):  # On ignore le label 0 (arrière-plan)
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
    
    # # Sauvegarde de l'image de prédiction et de l'image originale
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_np)
    # plt.title("Image Originale")
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(res_merge.cpu().numpy(), cmap='gray')
    # plt.title("Masques Prédits")
    # plt.axis('off')
    
    # plt.savefig(os.path.join(output_dir, f"Image_{i}.png"))
    # plt.close()
