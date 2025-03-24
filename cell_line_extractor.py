import os
import glob
import pandas as pd

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True) 

# Paramètres de filtrage
y_target = 320  # Exemple : fixer y au milieu de l'image
tolerance = 5  # Tolérance sur y (+/- 5 pixels)

# Chercher tous les fichiers "mask_measurements_*.csv"
csv_files = glob.glob(os.path.join(input_dir, "mask_measurements_*.csv"))

for input_csv in csv_files:
    # Extraire le nom du fichier sans le dossier
    file_name = os.path.basename(input_csv)
    file_name_filtered = file_name.replace(".csv", "_filtré.csv")

    # Définir le chemin de sortie
    output_csv = os.path.join(output_dir, file_name_filtered)

    # Charger le fichier CSV
    df = pd.read_csv(input_csv)

    # Filtrer les cellules selon y avec tolérance
    df_filtered = df[(df["Centroid_Y"] >= y_target - tolerance) & (df["Centroid_Y"] <= y_target + tolerance)]

    # Trier les cellules de droite à gauche (x décroissant)
    df_filtered = df_filtered.sort_values(by="Centroid_X", ascending=False)

    # Sauvegarder le fichier filtré
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Fichier filtré enregistré sous : {output_csv}")
