import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.spatial import KDTree
from utils import numerical_sort

# === Fonctions auxiliaires ===
def remove_near_duplicates(coords, threshold=10):
    tree = KDTree(coords)
    to_keep = []
    used = set()

    for i, pt in enumerate(coords):
        if i in used:
            continue
        neighbors = tree.query_ball_point(pt, r=threshold)
        used.update(neighbors)
        to_keep.append(i)

    return coords[to_keep]

def build_graph(coords, threshold):
    tree = KDTree(coords)
    pairs = tree.query_pairs(threshold)
    graph = {i: [] for i in range(len(coords))}
    for i, j in pairs:
        d = np.linalg.norm(coords[i] - coords[j])
        graph[i].append((j, d))
        graph[j].append((i, d))
    return graph

def find_linear_paths_linear_pref(coords, graph, distance_max=35, k=5, angle_thresh_deg=30):
    tree = KDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    paths = []
    angle_thresh_rad = np.radians(angle_thresh_deg)

    for idx in np.argsort(coords[:, 0]):
        if visited[idx]:
            continue

        path = [idx]
        visited[idx] = True
        current = idx
        prev_vector = None

        while True:
            distances, indices = tree.query(coords[current], k=k + 1)
            candidates = []

            for i, d in zip(indices[1:], distances[1:]):
                if visited[i] or d > distance_max:
                    continue
                direction = coords[i] - coords[current]
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue

                if len(path) >= 2:
                    prev_vector = coords[current] - coords[path[-2]]
                else:
                    prev_vector = direction

                prev_norm = np.linalg.norm(prev_vector)
                if prev_norm == 0:
                    continue

                cos_angle = np.dot(prev_vector, direction) / (prev_norm * norm)
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                if angle <= angle_thresh_rad:
                    candidates.append((i, d, angle))

            if not candidates:
                break

            candidates.sort(key=lambda x: (x[2], x[1]))
            next_idx = candidates[0][0]
            path.append(next_idx)
            visited[next_idx] = True
            current = next_idx

        if len(path) > 1:
            paths.append(path)

    return paths

# === Paramètres ===
input_dir = "/home/killian/sam2/inferences/15492"
output_dir = "/home/killian/sam2/Results/"
plots_dir = os.path.join(output_dir, "Connected_Cells", os.path.basename(os.path.normpath(input_dir)))
os.makedirs(plots_dir, exist_ok=True)

distance_threshold = 30
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
        all_coords = remove_near_duplicates(all_coords, threshold=10)

        graph = build_graph(all_coords, threshold=distance_threshold)
        linear_paths = find_linear_paths_linear_pref(all_coords, graph, distance_max=distance_threshold)

        # === Visualisation ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(binary, cmap="gray")
        axs[0].set_title("Masque Binaire")
        axs[0].axis("off")

        axs[1].imshow(binary, cmap="gray")
        axs[1].scatter(all_coords[:, 0], all_coords[:, 1], color='red', s=8)
        axs[1].set_title("Centroïdes connectés")
        axs[1].axis("off")
        for i, neighbors in graph.items():
            for j, _ in neighbors:
                axs[1].plot([all_coords[i][0], all_coords[j][0]], [all_coords[i][1], all_coords[j][1]], color="yellow", linewidth=0.5)

        axs[2].imshow(binary, cmap="gray")
        axs[2].set_title("Fils cellulaires linéaires")
        axs[2].axis("off")
        cmap = plt.get_cmap('tab20', len(linear_paths))
        for i, path in enumerate(linear_paths):
            pts = all_coords[path]
            color = cmap(i)
            axs[2].plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)
            axs[2].scatter(pts[:, 0], pts[:, 1], color=color, s=10)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{specimen}_{tile}_linear_paths.png"), dpi=300)
        plt.close()

        print(f"📸 Sauvegardé : {specimen}_{tile}_linear_paths.png")