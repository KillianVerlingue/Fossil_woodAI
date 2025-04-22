import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from utils import numerical_sort

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/13823/"
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
tolerance = 10  # Tolérance en pixels (plus utilisé ici)
radius = 40     # Rayon de recherche pour A*
gap_threshold = 20  # Seuil de distance pour détecter un hiatus

def a_star_cells(df_cells, start_idx, radius=40):
    """
    Recherche de proche en proche des cellules depuis un point de départ avec algo A*
    """
    coords = df_cells[["Centroid_X", "Centroid_Y"]].values
    tree = KDTree(coords)

    visited = set()
    path = []

    current_idx = start_idx
    path.append(current_idx)
    visited.add(current_idx)

    while True:
        current_point = coords[current_idx]
        idxs = tree.query_ball_point(current_point, r=radius)

        neighbors = [i for i in idxs if i not in visited and i != current_idx]
        if not neighbors:
            break

        # Optionnel : ajouter un critère directionnel ici
        next_idx = min(neighbors, key=lambda i: np.linalg.norm(coords[i] - current_point))

        visited.add(next_idx)
        path.append(next_idx)
        current_idx = next_idx

    return df_cells.iloc[path]

# Traitement des fichiers CSV
csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:
    df = pd.read_csv(input_csv)

    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue

    has_tile_id = "Tile_ID" in df.columns
    all_filtered_data = []
    unique_tiles = df["Tile_ID"].unique()

    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()

        # Supprimer les doublons de coordonnées
        df_tile = df_tile.sort_values(by="Area", ascending=False)
        df_tile = df_tile.drop_duplicates(subset=["Centroid_X", "Centroid_Y"], keep="first")
        df_tile = df_tile.sort_values(by="Centroid_X", ascending=False).reset_index(drop=True)

        if df_tile.empty:
            continue

        # A* : extraction de la file cellulaire
        df_astar = a_star_cells(df_tile, start_idx=0, radius=radius)
        df_astar["Tile_ID"] = tile

        # Calcul de l'épaisseur entre cellules
        thicknesses = []
        for i in range(len(df_astar) - 1):
            cell1 = df_astar.iloc[i]
            cell2 = df_astar.iloc[i + 1]

            centroid1 = np.array([cell1["Centroid_X"], cell1["Centroid_Y"]])
            centroid2 = np.array([cell2["Centroid_X"], cell2["Centroid_Y"]])
            distance = np.linalg.norm(centroid2 - centroid1) * 0.5220  # en µm

            diameter1 = cell1["Equivalent_Diameter"]
            diameter2 = cell2["Equivalent_Diameter"]

            if distance > gap_threshold:
                thickness = "NA"
            else:
                thickness = max(0, distance - (0.5 * (diameter1 + diameter2)))

            thicknesses.append(thickness)

        thicknesses.append(0)
        df_astar["2P_thickness"] = thicknesses
        all_filtered_data.append(df_astar)

        # Visualisation
        plt.figure(figsize=(8, 6))
        plt.scatter(df_tile["Centroid_X"], df_tile["Centroid_Y"],
                    color='gray', label="Cellules")
        plt.scatter(df_astar["Centroid_X"], df_astar["Centroid_Y"],
                    color='red', label="File cellulaire (A*)")
        plt.xlabel("Centroid_X")
        plt.ylabel("Centroid_Y")
        plt.title(f"File cellulaire (A*) - {tile} ({os.path.basename(input_csv)})")
        plt.legend()
        plt.gca().invert_yaxis()

        output_image = os.path.join(plots_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}.png')}")
        plt.savefig(output_image)
        plt.close()

    if all_filtered_data:
        all_filtered_data = sorted(all_filtered_data, key=lambda df: numerical_sort(df["Tile_ID"].iloc[0]))
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        output_csv = os.path.join(csv_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")
