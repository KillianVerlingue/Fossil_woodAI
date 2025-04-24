import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.spatial import KDTree
from utils import numerical_sort

# ==== PARAMÈTRES ====
input_dir = "/home/killian/sam2/inferences/15492"
output_dir = "/home/killian/sam2/Results/"
plots_dir = os.path.join(output_dir, "ContoursOnlyAdj", os.path.basename(os.path.normpath(input_dir)))
os.makedirs(plots_dir, exist_ok=True)

distance_threshold = 25  # Distance maximale entre centroïdes voisins
max_angle_deg = 45       # Angle maximal admissible entre segments successifs

# ==== FONCTIONS UTILITAIRES ====
def compute_angle(a, b, c):
    ab = b - a
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def greedy_path(start_idx, all_coords, tree, used, path, max_angle_deg):
    current_idx = start_idx
    used.add(current_idx)
    path.append(current_idx)
    prev_direction = None

    while True:
        neighbors = tree.query_ball_point(all_coords[current_idx], r=distance_threshold)
        neighbors = [n for n in neighbors if n != current_idx and n not in used]

        if not neighbors:
            break

        best_score = float('inf')
        best_neighbor = None

        for n in neighbors:
            direction = all_coords[n] - all_coords[current_idx]
            if prev_direction is not None:
                angle = angle_between_vectors(prev_direction, direction)
                if angle > max_angle_deg:
                    continue
            else:
                angle = 0

            distance = np.linalg.norm(direction)
            score = distance + angle * 0.1

            if score < best_score:
                best_score = score
                best_neighbor = n
                best_direction = direction

        if best_neighbor is None:
            break

        current_idx = best_neighbor
        prev_direction = best_direction
        used.add(current_idx)
        path.append(current_idx)

# ==== TRAITEMENT ====
csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:
    df = pd.read_csv(input_csv)
    specimen = os.path.splitext(os.path.basename(input_csv))[0].replace("mask_measurements_", "")
    print(f"\n🔍 Traitement : {specimen}")

    has_tile = "Tile_ID" in df.columns
    tiles = df["Tile_ID"].unique() if has_tile else [None]

    for tile in tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile else df.copy()

        if has_tile:
            df_tile = df_tile.sort_values(by="Area", ascending=False)
            df_tile = df_tile.drop_duplicates(subset=["Tile_ID", "Mask_ID"], keep="first")

        tif_name = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask_path = os.path.join(input_dir, "mask", tif_name)
        if not os.path.exists(mask_path):
            print(f"❌ Masque non trouvé : {mask_path}")
            continue

        mask = tifffile.imread(mask_path)
        binary = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        existing_coords = df_tile[["Centroid_X", "Centroid_Y"]].values.tolist()
        all_coords = existing_coords.copy()

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
        tree = KDTree(all_coords)

        used = set()
        cellular_files = []

        for idx in range(len(all_coords)):
            if idx not in used:
                file_cells = []
                greedy_path(idx, all_coords, tree, used, file_cells, max_angle_deg)
                if len(file_cells) > 2:
                    cellular_files.append(file_cells)

        # === Visualisation ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(binary, cmap="gray")
        axs[0].set_title("Masque Binaire")
        axs[0].axis("off")

        axs[1].imshow(np.zeros_like(binary), cmap="gray")
        for cnt in contours:
            axs[1].plot(cnt[:, 0, 0], cnt[:, 0, 1], color='lime', linewidth=1)
        axs[1].scatter(all_coords[:, 0], all_coords[:, 1], color='red', s=8, label="Centroïdes")

        for file_cells in cellular_files:
            for i in range(len(file_cells) - 1):
                x1, y1 = all_coords[file_cells[i]]
                x2, y2 = all_coords[file_cells[i + 1]]
                axs[1].plot([x1, x2], [y1, y2], color="cyan", linewidth=2)

            axs[1].scatter(all_coords[file_cells[0]][0], all_coords[file_cells[0]][1], color="orange", s=30)
            axs[1].scatter(all_coords[file_cells[-1]][0], all_coords[file_cells[-1]][1], color="blue", s=30)

        axs[1].set_title(f"Files cellulaires - {tile}")
        axs[1].axis("off")

        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f"{specimen}_{tile}_files_cells.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

        print(f"📸 Sauvegardé : {plot_file}")
