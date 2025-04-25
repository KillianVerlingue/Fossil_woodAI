import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.spatial import KDTree
from utils import numerical_sort
import networkx as nx
from sklearn.linear_model import LinearRegression

#fonction 

def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180

def is_aligned(p1, p2, ref_angle, tol=15):
    ang = angle_between(p1, p2)
    return abs((ang - ref_angle + 90) % 180 - 90) < tol


# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

# Créer les sous-dossiers plots et csv
plots_dir = os.path.join(output_dir, "Plots")
csv_dir = os.path.join(output_dir, "Data")

# Récupérer le nom du spécimen à partir du chemin d'entrée
specimen_name = os.path.basename(os.path.normpath(input_dir))

# Ajouter un sous-dossier spécifique au spécimen dans les répertoires plots et data
plots_dir = os.path.join(plots_dir, specimen_name)
csv_dir = os.path.join(csv_dir, specimen_name)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Paramètres
distance_threshold = 25  # distance max pour relier deux centroïdes

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

        # Chargement des masques
        tif_name = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask_path = os.path.join(input_dir, "mask", tif_name)

        mask = tifffile.imread(mask_path)
        binary = (mask > 0).astype(np.uint8)

        # Extraction des contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Centroïdes existants
        existing_coords = df_tile[["Centroid_X", "Centroid_Y"]].values.tolist()
        all_coords = existing_coords.copy()

        # Ajouter les centroïdes manquants à partir des contours
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

        files = []
        if len(all_coords) > 1:
            # Graphe de voisinage
            tree = KDTree(all_coords)
            pairs = tree.query_pairs(distance_threshold)

            G = nx.Graph()
            G.add_nodes_from(range(len(all_coords)))
            G.add_edges_from(pairs)

            # Orientation principale
            angles = [angle_between(all_coords[i], all_coords[j]) for i, j in G.edges]
            hist, bins = np.histogram(angles, bins=36, range=(0, 180))
            main_dir = bins[np.argmax(hist)]
            tolerance_angle = 15

            # Recherche de files cellulaires
            visited = set()
            for node in G.nodes:
                if node in visited:
                    continue
                chain = [node]
                queue = [node]
                visited.add(node)
                while queue:
                    current = queue.pop()
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited and is_aligned(all_coords[current], all_coords[neighbor], main_dir, tolerance_angle):
                            chain.append(neighbor)
                            queue.append(neighbor)
                            visited.add(neighbor)
                if len(chain) >= 3:
                    files.append(chain)

            # Scoring des files
            file_scores = []
            for idx_file, chain in enumerate(files):
                pts = np.array([all_coords[i] for i in chain])
                areas = []
                for i in chain:
                    x, y = all_coords[i]
                    mask_row = df_tile[(df_tile["Centroid_X"] == x) & (df_tile["Centroid_Y"] == y)]
                    if not mask_row.empty:
                        areas.append(mask_row.iloc[0]["Area"])

                # Longueur
                n_cells = len(chain)

                # Linéarité avec régression
                X = pts[:, 0].reshape(-1, 1)
                Y = pts[:, 1]
                reg = LinearRegression().fit(X, Y)
                r_squared = reg.score(X, Y)

                # Cohérence des aires
                area_std = np.std(areas) if areas else 700 # gros std 

                # Score composite
                norm_len = n_cells / max(1, max(len(f) for f in files))  # [0-1]
                score = (2 * r_squared) + (1 / (1 + area_std)) + (1.5 * norm_len)

                file_scores.append({
                    "File_ID": idx_file,
                    "n_cells": n_cells,
                    "linearity": round(r_squared, 3),
                    "area_std": round(area_std, 1),
                    "score": round(score, 3)
                })

            # Tri décroissant des files
            file_scores = sorted(file_scores, key=lambda x: x["score"], reverse=True)
            best_file_id = file_scores[0]["File_ID"]

        # === Plot : 3 visualisations
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Masque binaire
        axs[0].imshow(binary, cmap="gray")
        axs[0].set_title("Masque Binaire")
        axs[0].axis("off")

        # 2. Graphe de connexions
        axs[1].imshow(np.zeros_like(binary), cmap="gray")
        for cnt in contours:
            axs[1].plot(cnt[:, 0, 0], cnt[:, 0, 1], color='lime', linewidth=1)
        axs[1].scatter(all_coords[:, 0], all_coords[:, 1], color='red', s=8, label="Centroïdes")
        if len(all_coords) > 1:
            for i, j in pairs:
                x1, y1 = all_coords[i]
                x2, y2 = all_coords[j]
                axs[1].plot([x1, x2], [y1, y2], color="yellow", linewidth=0.8, alpha=0.7)
        axs[1].set_title("Centroïdes + Liens")
        axs[1].axis("off")

        # 3. Files cellulaires colorées
        axs[2].imshow(np.zeros_like(binary), cmap="gray")
        overlay = np.zeros((*binary.shape, 3), dtype=np.uint8)
        if files:
            # Définir des couleurs distinctes pour la meilleure file
            colors = (plt.cm.jet(np.linspace(0, 1, len(files)))[:, :3] * 255).astype(np.uint8)
            
            # Colorier toutes les files normalement
            for idx_file, chain in enumerate(files):
                for i in chain:
                    cX, cY = map(int, all_coords[i])
                    for cnt in contours:
                        if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0:
                            cv2.drawContours(overlay, [cnt], -1, color=tuple(int(c) for c in colors[idx_file]), thickness=cv2.FILLED)
                            break
            
            # Colorier la meilleure file de manière spéciale (en rouge)
            best_file_chain = files[best_file_id]
            for i in best_file_chain:
                cX, cY = map(int, all_coords[i])
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0:
                        cv2.drawContours(overlay, [cnt], -1, color=(255, 0, 0), thickness=cv2.FILLED)  # Rouge pour la meilleure file
                        break

        axs[2].imshow(cv2.addWeighted(cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2RGB), 0.3, overlay, 0.7, 0))
        axs[2].set_title("Files Cellulaires Colorées")
        axs[2].axis("off")

        # Enregistrement des données de la meilleure file
        if files:
            best_file_data = []
            best_chain = files[best_file_id]
            for cell_idx in best_chain:
                x, y = all_coords[cell_idx]
                best_file_data.append({"Tile_ID": tile, "File_ID": best_file_id, "Centroid_X": x, "Centroid_Y": y})
            
            df_best_file = pd.DataFrame(best_file_data)
            file_csv_path = os.path.join(csv_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}_best_file.csv')}")
            df_best_file.to_csv(file_csv_path, index=False)
            print(f"enregistrement de la meilleure file: {file_csv_path}")

        # Sauvegarde
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}_layout.png')}")
        plt.savefig(plot_file, dpi=300)
        plt.close()

        print(f"Fichier final enregistré sous : {plot_file}")
