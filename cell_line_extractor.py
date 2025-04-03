import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

# Créer les sous-dossiers plots et csv
plots_dir = os.path.join(output_dir, "Plots")
csv_dir = os.path.join(output_dir, "Data")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Paramètres
tolerance = 10 # Tolérance en pixels
N = 70         # Nombre de points à prendre à droite et à gauche
distance_threshold = 10 # Seuil de distance pour filtrer les cellules éloignées de la droite de régression
gap_threshold = 12  # Seuil de distance pour détecter un hiatus
num_lines = 50 # Nombre de droites régréssion 
max_slope = 0.3 # évite de prendre les diagonales
positive_slope_weight = 1.6 #pondère pour donner plus d'importance au pente positives 
# Chercher tous les fichiers "mask_measurements_*.csv"
csv_files = glob.glob(os.path.join(input_dir, "mask_measurements_*.csv"))

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

        # n = len(df_tile)
        # third = n // 3
        # df_tile["xquants"] = pd.qcut(df_tile["Centroid_X"],10,labels=False)
        # df_tile["yquants"] = pd.qcut(df_tile["Centroid_Y"],10,labels=False)
      
        # # Diviser les données en trois parties égales sur l'axe X avec Centroid_X"
        # df_first = df_tile.iloc[:third]
        # df_second = df_tile.iloc[third:2*third]
        # df_third = df_tile.iloc[2*third:]
        # # Ajuster une régression linéaire sur chaque segment (1/3, 2/3, et 3/3)
        # coeffs_first = np.polyfit(df_first["Centroid_X"], df_first["Centroid_Y"], 1)
        # coeffs_second = np.polyfit(df_second["Centroid_X"], df_second["Centroid_Y"], 1)
        # coeffs_third = np.polyfit(df_third["Centroid_X"], df_third["Centroid_Y"], 1)

        # # Extraire les coefficients pour chaque segment
        # a1, b1 = coeffs_first
        # a2, b2 = coeffs_second
        # a3, b3 = coeffs_third

        # # Créer un ensemble de points pour ajuster la droite finale
        # # Placer les droites de régression pour chaque segment
        # x_min, x_max = df_tile["Centroid_X"].min(), df_tile["Centroid_X"].max()
        # x_range = np.linspace(x_min, x_max, N)

        # # Calculer les valeurs Y pour chaque segment
        # y_first = a1 * x_range + b1
        # y_second = a2 * x_range + b2
        # y_third = a3 * x_range + b3

        # # Calculer la droite finale en ajustant pour minimiser l'écart entre les trois droites
        # # Cela revient à minimiser l'erreur entre les trois droites

        # # Calculer les erreurs 
        # error_first = np.sum((y_first - (a1 * x_range + b1))**2)
        # error_second = np.sum((y_second - (a2 * x_range + b2))**2)
        # error_third = np.sum((y_third - (a3 * x_range + b3))**2)

        # # Ajuster la droite finale (moyenne pondérée des trois droites)
        # final_a = (a1 + a2 + a3) / 3
        # final_b = (b1 + b2 + b3) / 3

        # # Calculer les valeurs Y pour la droite finale ajustée
        # y_final = final_a * x_range + final_b
       
        # # Ajuster une droite sur les coordonnées (Centroid_X, Centroid_Y) pour estimer la direction
        # coeffs = np.polyfit(df_tile["Centroid_X"], df_tile["Centroid_Y"], 1)
        # a, b = coeffs
       
        filtered_cells = []
        previous_x = None
        previous_y = None

        # Itérer sur les cellules en suivant la direction de la régression
        for _, row in df_tile.iterrows():
            current_x, current_y = row["Centroid_X"], row["Centroid_Y"]
            
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
            distance = np.sqrt((cell2["Centroid_X"] - cell1["Centroid_X"])**2 + (cell2["Centroid_Y"] - cell1["Centroid_Y"])**2)
            diameter1 = cell1["Equivalent_Diameter"]
            diameter2 = cell2["Equivalent_Diameter"]
           
            # Vérification du hiatus
            if distance > gap_threshold:
                thickness = "NA"
            else:
                thickness = abs(distance -(0.5 * diameter1) + distance -(0.5*diameter2))
            thicknesses.append(thickness)

        # Pour la dernière cellule, on met NaN pour signaler l'absence de calcul
        
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

        # # Tracer les trois droites de régression
        # plt.plot(x_range, y_first, color='red', label="Régression 1er tiers")
        # plt.plot(x_range, y_second, color='green', label="Régression 2e tiers")
        # plt.plot(x_range, y_third, color='blue', label="Régression 3e tiers")

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
        df_final = pd.concat(all_filtered_data, ignore_index=True)
        # Sauvegarde du fichier CSV 
        output_csv = os.path.join(csv_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
        df_final.to_csv(output_csv, index=False)
        print(f"Fichier final enregistré sous : {output_csv}")