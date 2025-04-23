import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from utils import numerical_sort

# === PARAMÈTRES ===
input_dir = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
neighbor_distance = 10
y_penalty = 0.7
max_vertical_shift = 10
min_group_size = 4

# === STRUCTURE ===
os.makedirs(output_dir, exist_ok=True)
plots_dir = os.path.join(output_dir, "Plots", os.path.basename(os.path.normpath(input_dir)))
csv_dir = os.path.join(output_dir, "Data", os.path.basename(os.path.normpath(input_dir)))
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"Fichier vide ignoré : {input_csv}")
        continue

    has_tile_id = "Tile_ID" in df.columns
    all_groups = []

    unique_tiles = df["Tile_ID"].unique() if has_tile_id else [None]
    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()
        df_tile = df_tile.sort_values(by="Area", ascending=False).drop_duplicates(subset=["Centroid_X", "Centroid_Y"], keep="first")

        coords = df_tile[["Centroid_X", "Centroid_Y"]].values
        G = nx.Graph()
        for i in range(len(coords)):
            G.add_node(i)

        # Agrégation stricte : droite ➝ gauche + faible ΔY
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i == j:
                    continue
                x1, y1 = coords[i]
                x2, y2 = coords[j]

                if x2 >= x1:
                    continue  # ❌ gauche uniquement
                if abs(y2 - y1) > max_vertical_shift:
                    continue  # ❌ trop de variation verticale

                dx = x2 - x1
                dy = (y2 - y1) * y_penalty
                dist = np.sqrt(dx**2 + dy**2)

                if dist < neighbor_distance:
                    G.add_edge(i, j, weight=dist)

        # Identifier les groupes
        components = [list(comp) for comp in nx.connected_components(G) if len(comp) >= min_group_size]
        color_map = plt.cm.get_cmap("tab20", len(components))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df_tile["Centroid_X"], df_tile["Centroid_Y"], color="lightgray", s=10, label="Cellules")

        for idx, comp in enumerate(components):
            group_df = df_tile.iloc[comp].copy()
            group_df["Group_ID"] = idx
            group_df["Tile_ID"] = tile
            all_groups.append(group_df)

            sorted_group = group_df.sort_values(by="Centroid_X", ascending=False)
            cx = sorted_group["Centroid_X"].values
            cy = sorted_group["Centroid_Y"].values
            plt.plot(cx, cy, color=color_map(idx), linewidth=1.5, label=f"Groupe {idx} ({len(group_df)})")
            plt.scatter(cx, cy, color=color_map(idx), edgecolor='k', s=12)

        plt.gca().invert_yaxis()
        plt.title(f"Agrégation dirigée - {tile} ({len(components)} groupes)")
        plt.xlabel("Centroid_X")
        plt.ylabel("Centroid_Y")
        plt.legend(fontsize=6, loc='upper right')
        outplot = os.path.join(plots_dir, f"{os.path.basename(input_csv).replace('.csv', f'_directional_{tile}.png')}")
        plt.tight_layout()
        plt.savefig(outplot, dpi=300)
        plt.close()

    # Export CSV
    if all_groups:
        df_final = pd.concat(all_groups, ignore_index=True)
        output_csv = os.path.join(csv_dir, os.path.basename(input_csv).replace(".csv", "_directional.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"✅ Groupes dirigés enregistrés : {output_csv}")
