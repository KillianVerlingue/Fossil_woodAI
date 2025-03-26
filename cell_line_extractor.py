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
tolerance = 10  # Tolérance en pixels
N = 50         # Nombre de points à prendre à droite et à gauche
distance_threshold = 15  # Seuil de distance pour filtrer les cellules éloignées de la droite de régression

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

        # Ajuster une droite sur les coordonnées (Centroid_X, Centroid_Y) pour estimer la direction
        coeffs = np.polyfit(df_tile["Centroid_X"], df_tile["Centroid_Y"], 1)
        a, b = coeffs  # Coefficients de la droite y = a * x + b

        # Trier les cellules par 'Centroid_X' de droite à gauche
        df_tile_sorted = df_tile.sort_values(by="Centroid_X", ascending=False)

        # Initialiser une liste pour les cellules filtrées
        filtered_cells = []
        previous_x = None  # Variable pour vérifier si la cellule suivante est sur le même X
        previous_y = None  # Variable pour suivre la coordonnée Y de la cellule précédente

        for _, row in df_tile_sorted.iterrows():
            current_x = row["Centroid_X"]
            current_y = row["Centroid_Y"]

            # Calculer la coordonnée Y théorique sur la droite de régression pour la position X donnée
            y_theorique = a * current_x + b

            # Calculer la distance entre la cellule et la droite de régression (distance absolue)
            distance_to_line = abs(current_y - y_theorique)

            # Si la distance est inférieure au seuil et que la cellule est suffisamment éloignée sur l'axe X
            if distance_to_line < distance_threshold:
                # Ajouter la cellule si elle est proche de la droite et suffisamment éloignée de la précédente
                if previous_x is None or abs(current_x - previous_x) > tolerance:
                    filtered_cells.append(row)
                    previous_x = current_x  # Mettre à jour le X précédent
                    previous_y = y_theorique  # Utiliser la coordonnée Y ajustée à la droite

        # Convertir les résultats en DataFrame
        df_filtered = pd.DataFrame(filtered_cells)

        # Ajouter l'identifiant de la tuile
        df_filtered["Tile_ID"] = tile

        # Ajouter aux résultats globaux
        all_filtered_data.append(df_filtered)

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
