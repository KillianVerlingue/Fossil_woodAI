import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
tolerance = 7  # Tolérance en pixels
N = 50         # Nombre de points à prendre à droite et à gauche
distance_threshold = 7  # Seuil de distance pour filtrer les cellules éloignées de la droite de régression
gap_threshold = 10  # Seuil de distance pour détecter un hiatus
num_lines = 20  # Nombre de droites de régression à tester

# Chercher tous les fichiers CSV
csv_files = glob.glob(os.path.join(input_dir, "mask_measurements_*.csv"))

for input_csv in csv_files:
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue

    has_tile_id = "Tile_ID" in df.columns
    all_filtered_data = []
    unique_tiles = df["Tile_ID"].unique() if has_tile_id else [None]

    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()
        df_tile = df_tile.sort_values(by="Centroid_X", ascending=False)

        best_line = None
        max_count = 0

        # Diviser les données en trois zones : haut, milieu, bas
        n = len(df_tile)
        thirds = n // 3
        zones = [df_tile.iloc[:thirds], df_tile.iloc[thirds:2*thirds], df_tile.iloc[2*thirds:]]

        for zone in zones:
            for _ in range(num_lines):
                x1, y1 = zone.sample(1)[["Centroid_X", "Centroid_Y"]].values[0]
                x2, y2 = zone.sample(1)[["Centroid_X", "Centroid_Y"]].values[0]
                if x1 == x2:
                    continue

                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1

                distances = np.abs(zone["Centroid_Y"] - (a * zone["Centroid_X"] + b))
                count = np.sum(distances < distance_threshold)

                if count > max_count:
                    max_count = count
                    best_line = (a, b)

        if best_line:
            a, b = best_line
            x_min, x_max = df_tile["Centroid_X"].min(), df_tile["Centroid_X"].max()
            x_range = np.linspace(x_min, x_max, N)
            y_best = a * x_range + b

            plt.figure(figsize=(8, 6))
            colors = ['red' if i > 0 and df_tile["Centroid_X"].iloc[i] < df_tile["Centroid_X"].iloc[i - 1] else 'gray' for i in range(len(df_tile))]
            plt.scatter(df_tile["Centroid_X"], df_tile["Centroid_Y"], c=colors, label="Cellules", alpha=0.7)
            plt.plot(x_range, y_best, color='purple', linestyle='--', label="Meilleure droite")
            plt.xlabel("Centroid_X")
            plt.ylabel("Centroid_Y")
            plt.title(f"Meilleure droite de régression - {tile} ({os.path.basename(input_csv)})")
            plt.legend()
            plt.gca().invert_yaxis()

            output_image = os.path.join(output_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}_best.png')}")
            plt.savefig(output_image)
            plt.close()

            print(f"Droite optimale enregistrée pour {tile} sous : {output_image}")

    if all_filtered_data:
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        output_csv = os.path.join(output_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")
