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
tolerance = 5  # Tolérance en pixels

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

    # Traitement par tuile
    if has_tile_id:
        unique_tiles = df["Tile_ID"].unique()
    else:
        unique_tiles = ["Global"]  # Si pas de tuiles définies, traiter tout ensemble

    for tile in unique_tiles:
        if has_tile_id:
            df_tile = df[df["Tile_ID"] == tile].copy()
        else:
            df_tile = df.copy()
        
        if df_tile.empty:
            continue

        # Ajuster une droite aux centroids pour capter l'orientation locale du bois
        coeffs = np.polyfit(df_tile["Centroid_X"], df_tile["Centroid_Y"], 1)  # y = ax + b
        a, b = coeffs

        # Calculer la distance de chaque point à la droite ajustée
        df_tile["distance"] = np.abs(df_tile["Centroid_Y"] - (a * df_tile["Centroid_X"] + b))

        # Filtrer les cellules proches de la ligne ajustée
        df_filtered = df_tile[df_tile["distance"] <= tolerance].copy()
        df_filtered.drop(columns=["distance"], inplace=True)

        # Ajouter l'identifiant de la tuile
        df_filtered["Tile_ID"] = tile

        # Trier les cellules filtrées de droite à gauche
        df_filtered = df_filtered.sort_values(by="Centroid_X", ascending=False)

        # Ajouter aux résultats globaux
        all_filtered_data.append(df_filtered)

        # Visualisation pour chaque tuile
        plt.figure(figsize=(8, 6))
        plt.scatter(df_tile["Centroid_X"], df_tile["Centroid_Y"], color='gray', label="Toutes les cellules", alpha=0.5)
        plt.scatter(df_filtered["Centroid_X"], df_filtered["Centroid_Y"], color='red', label="File cellulaire sélectionnée")

        # Tracer la ligne ajustée
        x_range = np.linspace(df_tile["Centroid_X"].min(), df_tile["Centroid_X"].max())
        y_range = a * x_range + b
        plt.plot(x_range, y_range, color='blue', label="Orientation du bois")

        plt.xlabel("Centroid_X")
        plt.ylabel("Centroid_Y")
        plt.title(f"File cellulaire - {tile} ({os.path.basename(input_csv)})")
        plt.legend()

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
