import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.spatial import KDTree
from utils import numerical_sort

# Dossiers
input_dir = "/home/killian/sam2/inferences/15492"
output_dir = "/home/killian/sam2/Results/"
plots_dir = os.path.join(output_dir, "Connected_Cells", os.path.basename(os.path.normpath(input_dir)))
os.makedirs(plots_dir, exist_ok=True)

# Paramètres
distance_threshold = 35  # distance max pour relier deux centroïdes

csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:

    df = pd.read_csv(input_csv)
    specimen = os.path.splitext(os.path.basename(input_csv))[0].replace("mask_measurements_", "")
    print(f"🔍 Traitement : {specimen}")

    has_tile = "Tile_ID" in df.columns
    tiles = df["Tile_ID"].unique() if has_tile else [None]

    for tile in tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile else df.copy()

        if has_tile:
            df_tile = df_tile.sort_values(by="Area", ascending=False)
            df_tile = df_tile.drop_duplicates(subset=["Tile_ID", "Mask_ID"], keep="first")

        # === Chargement du masque
        tif_name = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask_path = os.path.join(input_dir, "mask", tif_name)
        if not os.path.exists(mask_path):
            print(f"❌ Masque non trouvé : {mask_path}")
            continue

        mask = tifffile.imread(mask_path)
        binary = (mask > 0).astype(np.uint8)

        # === Extraction des contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === Centroïdes existants
        existing_coords = df_tile[["Centroid_X", "Centroid_Y"]].values.tolist()
        all_coords = existing_coords.copy()

        # === Ajouter centroïdes manquants à partir des contours
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            too_close = any(np.hypot(cx - x, cy - y) < 2 for x, y in existing_coords)
            if not too_close:
                all_coords.append([cx, cy])

        all_coords = np.array(all_coords)

        # === Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(binary, cmap="gray")
        axs[0].set_title("Masque Binaire")
        axs[0].axis("off")

        axs[1].imshow(np.zeros_like(binary), cmap="gray")  # Suppression fond binaire
        for cnt in contours:
            axs[1].plot(cnt[:, 0, 0], cnt[:, 0, 1], color='lime', linewidth=1)

        axs[1].scatter(all_coords[:, 0], all_coords[:, 1], color='red', s=8, label="Centroïdes")

        if len(all_coords) > 1:
            tree = KDTree(all_coords)
            pairs = tree.query_pairs(distance_threshold)
            for i, j in pairs:
                x1, y1 = all_coords[i]
                x2, y2 = all_coords[j]
                axs[1].plot([x1, x2], [y1, y2], color="yellow", linewidth=0.8, alpha=0.7)

        axs[1].set_title(f"Contours + Centroïdes + Liens - {tile}")
        axs[1].axis("off")

        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f"{specimen}_{tile}_contours_connected.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

        print(f"📸 Sauvegardé : {plot_file}")
