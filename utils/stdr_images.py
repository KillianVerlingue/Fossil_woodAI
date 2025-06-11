import tifffile
import cv2
import os
import glob
import numpy as np

scale_percent = 3.2  # Pourcentage de réduction
input_folder = "/home/killian/data2025"
output_folder = "/home/killian/data2025_cleaned"
os.makedirs(output_folder, exist_ok=True)

tif_files = glob.glob(os.path.join(input_folder, "**", "*.tif"), recursive=True)

for tif_path in tif_files:
    try:
        image = tifffile.imread(tif_path)

        # Gestion des images en niveaux de gris
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        height, width = image.shape[:2]
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_folder, os.path.basename(tif_path))
        tifffile.imwrite(output_path, resized)
        print(f"[✓] Redimensionné : {os.path.basename(tif_path)} → {new_width}x{new_height}")
    except Exception as e:
        print(f"[✗] Erreur sur {tif_path}: {e}")
