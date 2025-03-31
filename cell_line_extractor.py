import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/15485/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
tolerance = 10  # Tolérance en pixels
N = 50         # Nombre de points à prendre à droite et à gauche
distance_threshold = 15  # Seuil de distance pour filtrer les cellules éloignées de la droite de régression
gap_threshold = 15  # Seuil de distance pour détecter un hiatus

# Chercher tous les fichiers "mask_measurements_*.csv"
csv_files = glob.glob(os.path.join(input_dir, "mask_measurements_*.csv"))

for input_csv in csv_files:
    # Charger le fichier CSV
    df = pd.read_csv(input_csv)
    
    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue

    # Vérifier si la colonne 'Tile_ID' existe (sinon, on suppose une seule tuile)
    has_tile_id = "Tile_ID" in df.columns

    # Stocker les résultats de toutes les tuiles
    all_filtered_data = []
    unique_tiles = df["Tile_ID"].unique()

    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()
        if df_tile.empty:
            continue

        # Ajuster une droite sur les coordonnées (Centroid_X, Centroid_Y) pour estimer la direction
        coeffs = np.polyfit(df_tile["Centroid_X"], df_tile["Centroid_Y"], 1)
        a, b = coeffs
        df_tile_sorted = df_tile.sort_values(by="Centroid_X", ascending=False)

        filtered_cells = []
        previous_x = None
        previous_y = None

        for _, row in df_tile_sorted.iterrows():
            current_x, current_y = row["Centroid_X"], row["Centroid_Y"]
            y_theorique = a * current_x + b
            distance_to_line = abs(current_y - y_theorique)

            if distance_to_line < distance_threshold:
                if previous_x is None or abs(current_x - previous_x) > tolerance:
                    filtered_cells.append(row)
                    previous_x, previous_y = current_x, y_theorique

        df_filtered = pd.DataFrame(filtered_cells)
        df_filtered["Tile_ID"] = tile

        # Calcul du 2P_thickness avec gestion des hiatus
        thicknesses = []
        for i in range(len(df_filtered) - 1):
            cell1 = df_filtered.iloc[i]
            cell2 = df_filtered.iloc[i + 1]
            distance = np.sqrt((cell2["Centroid_X"] - cell1["Centroid_X"])**2 + (cell2["Centroid_Y"] - cell1["Centroid_Y"])**2)
            diameter1 = cell1["Equivalent_Diameter"]
            diameter2 = cell2["Equivalent_Diameter"]
            if distance > gap_threshold:
                thickness = 0
            else:
                thickness = distance - 0.5 * (diameter1 + diameter2)
            thicknesses.append(thickness)

        thicknesses.append(0)
        df_filtered["2P_thickness"] = thicknesses
        all_filtered_data.append(df_filtered)

    if all_filtered_data:
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        output_csv = os.path.join(output_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")


        # Visualisation
        plt.figure(figsize=(8, 6))
        # Toutes les cellules en gris
        plt.scatter(df_tile_sorted["Centroid_X"], df_tile_sorted["Centroid_Y"],
                    color='gray', label="Cellules")
        # Cellules filtrées en rouge
        plt.scatter(df_filtered["Centroid_X"], df_filtered["Centroid_Y"],
                    color='red', label="Cellules dans la file")

        # Tracer la droite de régression
        x_min, x_max = df_tile_sorted["Centroid_X"].min(), df_tile_sorted["Centroid_X"].max()
        x_range = np.linspace(x_min, x_max, 200)
        y_range = a * x_range + b
        plt.plot(x_range, y_range, color='blue', label="Direction estimée")

        plt.xlabel("Centroid_X")
        plt.ylabel("Centroid_Y")
        plt.title(f"File cellulaire - {tile} ({os.path.basename(input_csv)})")
        plt.legend()
        plt.gca().invert_yaxis() 

        # Sauvegarde de l'image
        output_image = os.path.join(output_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}.png')}")
        plt.savefig(output_image)
        plt.close()

    # Concaténer toutes les tuiles pour la grande image
    if all_filtered_data:
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        output_csv = os.path.join(output_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")
