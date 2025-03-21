import os
import numpy as np
import pandas as pd
import csv

# Paramètres
path="/home/killian/sam2/inferences2/mask_measurements.csv"
input_csv = path
output_csv = "/home/killian/sam2/inferences2/mask_measurements_filtré.csv"
y_target = 200  # Exemple : fixer y au milieu de l'image
threshold = 1  # Tolérance sur y (+/- 5 pixels)

# Charger le fichier CSV
df = pd.read_csv(input_csv)

# Filtrer les cellules selon y avec tolérance
df_filtered = df[(df["Centroid_Y"] >= y_target - threshold) & (df["Centroid_Y"] <= y_target + threshold)]

# Trier les cellules de droite à gauche (x décroissant)
df_filtered = df_filtered.sort_values(by="Centroid_X", ascending=False)

# Sauvegarder le fichier filtré
df_filtered.to_csv(output_csv, index=False)

print(f"Fichier filtré enregistré sous : {output_csv}")
