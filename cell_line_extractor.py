import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import numerical_sort

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/13/"
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
tolerance = 10 # Tolérance en pixels
N = 70         # Nombre de points à prendre à droite et à gauche
distance_threshold = 10 # Seuil de distance pour filtrer les cellules éloignées de la droite de régression
gap_threshold = 20  # Seuil de distance pour détecter un hiatus
num_lines = 50 # Nombre de droites régréssion 
max_slope = 0.3 # évite de prendre les diagonales
positive_slope_weight = 1.6 #pondère pour donner plus d'importance au pente positives 
# max_area = 28 #filtre les cellules abérantes selon le seuil en µm

# Chercher tous les fichiers "mask_measurements_*.csv"
csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:
    # Charger le fichier CSV
    df = pd.read_csv(input_csv)
    
    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue

    # Vérifier si la colonne 'Tile_ID' existe (sinon, on charge tout)
    has_tile_id = "Tile_ID" in df.columns
   
    # Stocker les résultats de toutes les tuiles
    all_filtered_data = []
    unique_tiles = df["Tile_ID"].unique()

    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()

    # Supprimer les doublons de coordonnées, en gardant celui avec la plus grande area
        df_tile = df_tile.sort_values(by="Area", ascending=False)
        df_tile = df_tile.drop_duplicates(subset=["Centroid_X", "Centroid_Y"], keep="first")

        # Trier pour l'analyse suivante
        df_tile = df_tile.sort_values(by="Centroid_X", ascending=False)

        best_line = None
        max_count = 0

        for _ in range(num_lines):
            x1, y1 = df_tile.sample(1)[["Centroid_X", "Centroid_Y"]].values[0]
            x2, y2 = df_tile.sample(1)[["Centroid_X", "Centroid_Y"]].values[0]
            if x1 == x2:
                continue

            a = (y2 - y1) / (x2 - x1)

            # Filtrer les pentes trop raides
            if abs(a) > max_slope:
                continue
            # Pondérer pour favoriser les pentes positives
            if a > 0:
                count_weighted = positive_slope_weight
            else:
                count_weighted = 1
            b = y1 - a * x1

            distances = np.abs(df_tile["Centroid_Y"] - (a * df_tile["Centroid_X"] + b))
            count = np.sum(distances < distance_threshold)

            if count > max_count:
                max_count = count
                best_line = (a, b)

        if best_line:
            a, b = best_line
            x_min, x_max = df_tile["Centroid_X"].min(), df_tile["Centroid_X"].max()
            x_range = np.linspace(x_min, x_max, N)
            y_best = a * x_range + b

        filtered_cells = []
        previous_x = None
        previous_y = None

        # Itérer sur les cellules en suivant la direction de la régression
        for _, row in df_tile.iterrows():
            current_x, current_y = row["Centroid_X"], row["Centroid_Y"]
            area = row["Area"]  # Assurez-vous que la colonne 'Area' existe dans votre DataFrame
            # area_um = area * (0.5220 ** 2)  # Conversion en µm²

            # # Vérification de la taille de la cellule
            # if area_um > max_area:
            #     continue  # Ignorer les petites cellules

            # Calculer la position théorique de Y en fonction de la droite finale ajustée
            y_theorique = a * current_x + b
            
            # Calculer la distance entre la cellule et la droite théorique
            distance_to_line = abs(current_y - y_theorique)
            
            # Si la cellule est proche de la droite, et qu'elle respecte la tolérance de distance
            if distance_to_line < distance_threshold:
                if previous_x is None or abs(current_x - previous_x) > tolerance:
                    filtered_cells.append(row)
                    previous_x, previous_y = current_x, y_theorique

        df_filtered = pd.DataFrame(filtered_cells)
        df_filtered["Tile_ID"] = tile
        all_filtered_data.append(df_filtered)

        # Extraction épaisseur des parois avec gestion des hiatus
        thicknesses = []

        for i in range(len(df_filtered) - 1):
            cell1 = df_filtered.iloc[i]
            cell2 = df_filtered.iloc[i + 1]
            
            # Calcul de la distance entre les centroides
            centroid1 = np.array([cell1["Centroid_X"], cell1["Centroid_Y"]])
            centroid2 = np.array([cell2["Centroid_X"], cell2["Centroid_Y"]])
            distance = np.linalg.norm(centroid2 - centroid1) * 0.5220  # Conversion en µm

            diameter1 = cell1["Equivalent_Diameter"]
            diameter2 = cell2["Equivalent_Diameter"]
            
            # Vérification du hiatus
            if distance > gap_threshold:
                thickness = "NA"
            else:
                # Calcul correct de l'épaisseur de la double paroi
                thickness = max(0, distance - (0.5 * (diameter1 + diameter2)))
                
            thicknesses.append(thickness)


        # Pour la dernière cellule, on met Na pour signaler l'absence de calcul
        thicknesses.append(0)  # Pour la dernière cellule on met 0 mais à rettiré si on travail sur bande
        df_filtered["2P_thickness"] = thicknesses


      # Visualisation
        plt.figure(figsize=(8, 6))
        # Toutes les cellules en gris
        plt.scatter(df_tile["Centroid_X"], df_tile["Centroid_Y"],
                    color='gray', label="Cellules")
        # Cellules filtrées en rouge
        plt.scatter(df_filtered["Centroid_X"], df_filtered["Centroid_Y"],
                    color='red', label="Cellules dans la file")
        # Tracer la droite finale ajustée
        plt.plot(x_range, y_best, color='purple', linestyle='--', label="Régression finale ajustée")
        plt.xlabel("Centroid_X")
        plt.ylabel("Centroid_Y")
        plt.title(f"File cellulaire - {tile} ({os.path.basename(input_csv)})")
        plt.legend()
        plt.gca().invert_yaxis()
        
        # Sauvegarde du Plot
        output_image = os.path.join(plots_dir, f"{os.path.basename(input_csv).replace('.csv', f'_{tile}.png')}")
        plt.savefig(output_image)
        plt.close()

    if all_filtered_data:
        all_filtered_data = sorted(all_filtered_data, key=lambda df: numerical_sort(df["Tile_ID"].iloc[0]))
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        # Sauvegarde du fichier CSV 
        output_csv = os.path.join(csv_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")