import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize
import cv2
import tifffile
from scipy.spatial import KDTree
from utils import numerical_sort
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

# Paramètres
distance_threshold = 40 # distance max pour relier deux centroïdes
tolerance_angle = 8 #tolérance de variation d'angles dans la recherhe de file
score_nb = 0.8 #importance en % du nombre de cellule dans la meilleure file cellulaire
score_area = 0.1 #importance en % de la variations d'aires dans la meilleure file cellulaire
score_angle =  0.1 #importance en % de la variations d'angles dans la meilleure file cellulaire
cells_per_tiles = 30 #nombre estimé de cellules par tuiles (droites à gauches)


# Dossiers d'entrée et de sortie
# input_dir = "/home/killian/sam2/inferences/15485/"
input_dir = "/home/killian/sam2/inferences/15492/"
# input_dir = "/home/killian/sam2/inferences/13823/"
# input_dir = "/home/killian/sam2/inferences/TGV4/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

####################### DEFINITIONS DES FONCTIONS #########################

# Fonction pour calculer l'angle entre deux points
def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180

# Fonction pour vérifier l'alignement des cellules
def is_aligned(p1, p2, ref_angle, tol=15):
    ang = angle_between(p1, p2)
    return abs((ang - ref_angle + 90) % 180 - 90) < tol

def score_file(chain, coords, df_tile):
    n_cells = len(chain)
    areas = []
    eq_diams = []
    for cell_idx in chain:
        x, y = coords[cell_idx]
        row = df_tile[(df_tile["Centroid_X"] == x) & (df_tile["Centroid_Y"] == y)]
        if not row.empty:
            areas.append(row.iloc[0]["Area"])
            eq_diams.append(row.iloc[0]["Equivalent_Diameter"])

    if len(areas) == 0:
        return 0  # ou un score minimal plus explicite que -inf

    areas = np.array(areas)
    mean_area = np.mean(areas)
    area_std = np.std(areas) / mean_area if mean_area != 0 else 0

    angle_devs = []
    for i in range(1, len(chain)):
        angle = angle_between(coords[chain[i-1]], coords[chain[i]])
        angle_devs.append(angle)

    angle_var = np.std(angle_devs) / 180 if len(angle_devs) > 1 else 0

    score = (
        (n_cells / cells_per_tiles) * score_nb +
        (1 - area_std) * score_area +
        (1 - angle_var) * score_angle
    )

    return score


# Mesurer la disatnce de cellules a son contour
def distance_to_contour_along_direction(centroid, direction, contour):
    direction = direction / np.linalg.norm(direction)
    min_proj_dist = np.inf

    for pt in contour[:, 0, :]:  # contour.shape = (N, 1, 2)
        vec = pt - centroid
        proj_length = np.dot(vec, direction)
        orth_dist = np.linalg.norm(vec - proj_length * direction)

        # On cherche les points alignés avec la direction (tolérance angulaire)
        if orth_dist < 1.0:  # tolérance faible => presque aligné
            if proj_length > 0:  # éviter de prendre des points derrière
                min_proj_dist = min(min_proj_dist, proj_length)

    return min_proj_dist if min_proj_dist != np.inf else 0

def apply_watershed(binary_mask):
    # Nettoyage initial
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform pour trouver les centres de cellules
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.07 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Labellisation des composantes
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # pour ne pas avoir de conflit avec l’arrière-plan
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR), markers)
    segmented = np.zeros_like(binary_mask)

    # Marquer les régions segmentées
    segmented[markers > 1] = 1

    return segmented.astype(np.uint8)

################ CHEMINs & DOSSIERS ################

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

csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

######################## PRINCIPAL ##########################

# Création d'un dataframe vide pour collecter toutes les données
columns = ["Tile_ID", "Nb_Cells", "Area", "Equivalent_Diameter", "Centroid_X", "Centroid_Y", "2P_Thickness"]
final_results_df = pd.DataFrame(columns=columns)

for input_csv in csv_files:
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue

    # Vérifier si la colonne 'Tile_ID' existe (sinon, on charge tout)
    has_tile_id = "Tile_ID" in df.columns
    tiles = df["Tile_ID"].unique() if has_tile_id else [None]
    # Stocker les résultats de toutes les tuiles
    all_filtered_data = []
    unique_tiles = df["Tile_ID"].unique()
    
    specimen = os.path.splitext(os.path.basename(input_csv))[0].replace("mask_measurements_", "")
    print(f"Traitement : {specimen}")

    summary_data = []

    for tile in unique_tiles:
        df_tile = df[df["Tile_ID"] == tile].copy() if has_tile_id else df.copy()

        if has_tile_id:
            df_tile = df_tile.sort_values(by="Area", ascending=False)
            df_tile = df_tile.drop_duplicates(subset=["Tile_ID", "Mask_ID"], keep="first")

        # Chargement des masques
        tif_name = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask_path = os.path.join(input_dir, "mask", tif_name)
        mask = tifffile.imread(mask_path)
        binary = (mask > 0).astype(np.uint8)
        mask = tifffile.imread(mask_path)

        # wateshed
        binary = apply_watershed(binary)

        # Extraction des contours
        binary = np.int32(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
        # Centroïdes existants
        existing_coords = df_tile[["Centroid_X", "Centroid_Y"]].values.tolist()
        # all_coords = existing_coords.copy()
        all_coords = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroid = np.array([cx, cy])

            # Trouver tous les centroïdes existants proches du contour
            matching = []
            for ec in existing_coords:
                dist = cv2.pointPolygonTest(cnt, (ec[0], ec[1]), measureDist=True)
                if dist >= -2:  # autoriser un petit dépassement
                    matching.append(ec)

            if len(matching) == 0:
                # Aucun centroïde : on ajoute le centroïde calculé à partir du contour
                all_coords.append([cx, cy])
            elif len(matching) == 1:
                # Un seul centroïde : on le garde
                all_coords.append(matching[0])
            else:
                # Plusieurs centroïdes dans un même contour : on les regroupe (barycentre)
                avg = np.mean(matching, axis=0)
                all_coords.append(avg)

        all_coords = np.array(all_coords)

        files = []
        if len(all_coords) > 1:
            tree = KDTree(all_coords)
            pairs = tree.query_pairs(distance_threshold)

            G = nx.Graph()
            G.add_nodes_from(range(len(all_coords)))
            G.add_edges_from(pairs)

            angles = [angle_between(all_coords[i], all_coords[j]) for i, j in G.edges]
            hist, bins = np.histogram(angles, bins=36, range=(0, 180))
            main_dir = bins[np.argmax(hist)]

            visited = set()
            for node in G.nodes:
                if node in visited:
                    continue
                chain = [node]
                queue = [node]
                visited.add(node)
                while queue:
                    current = queue.pop()
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited and is_aligned(all_coords[current], all_coords[neighbor], main_dir, tolerance_angle):
                            chain.append(neighbor)
                            queue.append(neighbor)
                            visited.add(neighbor)
                if len(chain) >= 3:
                    files.append(chain)

            # Sélection de la meilleure file :
            file_scores = [score_file(chain, all_coords, df_tile) for chain in files]
            best_file_id = np.argmax(file_scores)

        results_per_image = []

        # Enregistrement des résultats visuels
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(binary, cmap="gray")
        axs[0].set_title("Masque Binaire")
        axs[0].axis("off")

        axs[1].imshow(np.zeros_like(binary), cmap="gray")
        for cnt in contours:
            axs[1].plot(cnt[:, 0, 0], cnt[:, 0, 1], color='lime', linewidth=1)
        axs[1].scatter(all_coords[:, 0], all_coords[:, 1], color='red', s=8)
        axs[1].set_title("Centroïdes + Liens")
        axs[1].axis("off")

        axs[2].imshow(np.zeros_like(binary), cmap="gray")
        overlay = np.zeros((*binary.shape, 3), dtype=np.uint8)

        if files:
            # Génère une colormap pour N files
            file_scores = [score_file(chain, all_coords, df_tile) for chain in files]
            norm = Normalize(vmin=min(file_scores), vmax=max(file_scores))
            cmap = colormaps["coolwarm"]

            for idx_file, chain in enumerate(files):
                # Convertir couleur RGBA → BGR (OpenCV) avec des entiers
                rgba = cmap(norm(file_scores[idx_file]))[:3]
                color = tuple(int(255 * c) for c in rgba[::-1])  # RGB → BGR

                for i in chain:
                    cX, cY = map(int, all_coords[i])
                    for cnt in contours:
                        if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0:
                            cv2.drawContours(overlay, [cnt], -1, color=color, thickness=cv2.FILLED)
                            break
        # Colorer la meilleure file en rouge
        best_file_chain = files[best_file_id]
        best_coords = []

        for i in best_file_chain:
            cX, cY = map(int, all_coords[i])
            best_coords.append((cX, cY))
            for cnt in contours:
                if cv2.pointPolygonTest(cnt, (cX, cY), False) >= 0:
                    cv2.drawContours(overlay, [cnt], -1, color=(255, 0, 0), thickness=cv2.FILLED)

        # Tracer une ligne reliant les centroïdes de la meilleure file
        best_coords = np.array(best_coords)
        axs[2].plot(best_coords[:, 0], best_coords[:, 1], color="red", linewidth=2, marker='o')

        axs[2].imshow(cv2.addWeighted(cv2.cvtColor(np.uint8(binary * 255), cv2.COLOR_GRAY2RGB), 0.3, overlay, 0.7, 0))
        axs[2].set_title("Files Cellulaires Colorées")
        axs[2].axis("off")


        # Chemin du fichier à enregistrer
        plot_filename = f"{specimen}_{tile}_plot.png" if tile else f"{specimen}_plot.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)  # Pour libérer la mémoire

    # Enregistrement des résultats de la tuile
    results_per_tile = []
    if files:
       best_file_data = []
    for idx in best_file_chain:
        cx, cy = all_coords[idx]
        
        # Récupérer la ligne correspondante dans df_tile (avec tolérance si besoin)
        matched_row = df_tile[
            (np.isclose(df_tile["Centroid_X"], cx, atol=1.0)) &
            (np.isclose(df_tile["Centroid_Y"], cy, atol=1.0))
        ]
        
        if not matched_row.empty:
            row = matched_row.iloc[0]
            best_file_data.append({
                "Tile_ID": tile,
                "file_id": best_file_id,
                "score_file": file_scores[best_file_id],
                "Cell_ID": row.get("Mask_ID", idx),  # ou un autre identifiant si disponible
                "Area": row["Area"],
                "Equivalent_Diameter": row["Equivalent_Diameter"],
                "Centroid_X": row["Centroid_X"],
                "Centroid_Y": row["Centroid_Y"],
                "2P_Thickness": np.nan 
            })

    results_per_tile.extend(best_file_data)

    # Ajouter les résultats de cette tuile au dataframe global
    if results_per_tile:
        df_tile = pd.DataFrame(results_per_tile)
        final_results_df = pd.concat([final_results_df, df_tile], ignore_index=True)


# Écriture d'un seul fichier CSV final par image (PB1 ou PB2)
output_csv = os.path.join(csv_dir, os.path.basename(input_csv).replace(".csv", "_final.csv"))
final_results_df.to_csv(output_csv, index=False)

print(f"Fichier final par image sauvegardé : {output_csv}")
