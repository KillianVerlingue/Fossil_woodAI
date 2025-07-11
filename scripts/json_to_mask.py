import json
import numpy as np
import cv2
import os
import fnmatch
from pathlib import Path

root_dir = '/path/to/dir'

# Définition des couleurs pour chaque classe
CLASS_COLORS = {
    "T": (255, 0, 0),  # Bleu
    "R": (0, 255, 0),  # Vert
    "Bulle": (0, 0, 255)  # Rouge
}

def gen_mask_img(json_filename):
    # Lire le fichier JSON
    with open(json_filename, "r") as f:
        data = json.load(f)
    
    target_dirname = os.path.dirname(json_filename)
    
    # Créer une image de masque vierge
    mask = np.zeros((640, 640, 3), dtype=np.uint8) # format de l'image 640 par 640 et 3 pour le nombre de canaux

    # Boucle sur toutes les formes présentes dans le JSON
    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)  # Conversion en int32
        label = shape["label"]  # Récupération du label de l'objet

        # Vérifier si la classe est définie dans notre dictionnaire
        color = CLASS_COLORS.get(label, (255, 255, 255))  # Blanc par défaut

        # Dessiner la forme sur le masque
        cv2.fillPoly(mask, [points], color)
    
    # Sauvegarde du masque
    json_filename_noex = Path(json_filename).stem
    mask_img_filename = os.path.join(target_dirname, json_filename_noex + "_mask.png")
    cv2.imwrite(mask_img_filename, mask)

def main():
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_filepath = os.path.join(dirname, filename)
                print("Génération du masque pour :", json_filepath)
                gen_mask_img(json_filepath)

if __name__ == '__main__':
    main()
