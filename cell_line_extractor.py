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
    
    # Ajuster une droite aux centroids pour capter l'orientation locale du bois
    coeffs = np.polyfit(df["Centroid_X"], df["Centroid_Y"], 1)  # y = ax + b
    a, b = coeffs
    
    # Calculer la distance de chaque point à la droite ajustée
    df["distance"] = np.abs(df["Centroid_Y"] - (a * df["Centroid_X"] + b))
    
    # Filtrer les cellules proches de la ligne ajustée
    df_filtered = df[df["distance"] <= tolerance].copy()
    df_filtered = df_filtered.drop(columns=["distance"])
    
    # Trier les cellules filtrées de droite à gauche
    df_filtered = df_filtered.sort_values(by="Centroid_X", ascending=False)
    
    # Enregistrer le fichier filtré
    file_name = os.path.basename(input_csv)
    file_name_filtered = file_name.replace(".csv", "_filtré.csv")
    output_csv = os.path.join(output_dir, file_name_filtered)
    df_filtered.to_csv(output_csv, index=False)
    
    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Centroid_X"], df["Centroid_Y"], color='gray', label="Toutes les cellules", alpha=0.5)
    plt.scatter(df_filtered["Centroid_X"], df_filtered["Centroid_Y"], color='red', label="File cellulaire sélectionnée")
    
    # Tracer la ligne ajustée
    x_range = np.linspace(df["Centroid_X"].min(), df["Centroid_X"].max())
    y_range = a * x_range + b
    plt.plot(x_range, y_range, color='blue', label="Orientation du bois")
    
    plt.xlabel("Centroid_X")
    plt.ylabel("Centroid_Y")
    plt.title(f"File cellulaire - {file_name}")
    plt.legend()
    
    # Sauvegarde de l'image
    output_image = os.path.join(output_dir, file_name.replace(".csv", ".png"))
    plt.savefig(output_image)
    plt.close()
    
    print(f"Fichier filtré enregistré sous : {output_csv}")
    print(f"Image enregistrée sous : {output_image}")