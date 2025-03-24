import os
import glob
import pandas as pd
import numpy as np

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

    # Trier de droite à gauche pour bien identifier la file cellulaire
    df = df.sort_values(by="Centroid_X", ascending=False)

    # Ajuster une droite aux données
    coeffs = np.polyfit(df["Centroid_X"], df["Centroid_Y"], 1)  # y = ax + b
    a, b = coeffs

    # Filtrer les cellules proches de la ligne ajustée
    df["distance"] = np.abs(df["Centroid_Y"] - (a * df["Centroid_X"] + b))
    df_filtered = df[df["distance"] <= tolerance]

    # Supprimer la colonne de distance
    df_filtered = df_filtered.drop(columns=["distance"])

    # Enregistrer le fichier filtré
    file_name = os.path.basename(input_csv)
    file_name_filtered = file_name.replace(".csv", "_filtré.csv")
    output_csv = os.path.join(output_dir, file_name_filtered)
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Fichier filtré enregistré sous : {output_csv}")

