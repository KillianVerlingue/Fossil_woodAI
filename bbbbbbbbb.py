import os
import glob
import pandas as pd
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
import networkx as nx
from utils import numerical_sort

# Paramètres ajustables
distance_threshold = 45   # distance max pour relier deux centroïdes
tolerance_angle   = 12   # tolérance d'angle (°)
Sblt              = 0.02 # seuil watershed
score_nb          = 0.7  # poids nombre de cellules
score_area        = 0.1  # poids variation d'aires
score_angle       = 0.2  # poids variation d'angles
cells_per_tile    = 40   # normalisation nombre de cellules

# Dossiers
input_dir  = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
plots_dir  = os.path.join(output_dir, "Plots")
csv_dir    = os.path.join(output_dir, "Data")

specimen_name = os.path.basename(os.path.normpath(input_dir))
plots_dir  = os.path.join(plots_dir, specimen_name)
csv_dir    = os.path.join(csv_dir, specimen_name)
for d in [output_dir, plots_dir, csv_dir]:
    os.makedirs(d, exist_ok=True)

# Fonctions utilitaires

def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180

def is_aligned(p1, p2, ref_angle, tol=tolerance_angle):
    ang = angle_between(p1, p2)
    return abs((ang - ref_angle + 90) % 180 - 90) < tol

def score_file(chain, coords, df_tile):
    # Collecte aires pour évaluer uniformité
    areas = []
    for idx in chain:
        x, y = coords[idx]
        row = df_tile[(np.isclose(df_tile['Centroid_X'], x, atol=1)) \
                     & (np.isclose(df_tile['Centroid_Y'], y, atol=1))]
        if not row.empty:
            areas.append(row.iloc[0]['Area'])
    if len(chain) < 2 or not areas:
        return 0.0
    mean_area = np.mean(areas)
    area_std  = np.std(areas) / mean_area if mean_area else 0
    area_score = 1 - min(area_std, 1.0)

    # Variation d'angle
    angle_devs = [angle_between(coords[chain[i-1]], coords[chain[i]]) \
                  for i in range(1, len(chain))]
    angle_var  = np.std(angle_devs) / 180 if len(angle_devs) > 1 else 0
    angle_score = 1 - min(angle_var, 1.0)

    # Longueur de la file
    length_score = min(len(chain) / cells_per_tile, 1.0)

    score = (score_nb * length_score +
             score_area * area_score +
             score_angle * angle_score)
    return round(score, 4)

def apply_watershed(binary_mask):
    # Fermer les petites ouvertures pour récupérer meilleurs contours
    kernel  = np.ones((3,3), np.uint8)
    closed  = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, Sblt * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(binary_mask * 255, \
                                         cv2.COLOR_GRAY2BGR), markers)
    seg = np.zeros_like(binary_mask)
    seg[markers > 1] = 1
    return seg

# Lecture des CSV et mapping tiles numériques séquentiels
csv_files = sorted(glob.glob(os.path.join(input_dir, 'mask_measurements_*.csv')), key=numerical_sort)
for csv_path in csv_files:
    df_all = pd.read_csv(csv_path)
    if df_all.empty:
        continue
    specimen = os.path.basename(csv_path).replace('mask_measurements_','').replace('.csv','')

    # Créer mapping tile: Image_0->1, Image_1->2, ...
    unique_tiles = sorted(df_all['Tile_ID'].unique(), key=numerical_sort)
    tile_map = {t: i+1 for i, t in enumerate(unique_tiles)}

    all_best = []
    for tile in unique_tiles:
        df_tile = df_all[df_all['Tile_ID'] == tile].copy()
        # Filtrer doublons, garder plus grande aire
        df_tile = df_tile.sort_values('Area', ascending=False) \
                         .drop_duplicates(subset=['Tile_ID','Mask_ID'])

        # Chargement et segmentation masque
        fname = f"{specimen}_{tile}_mask.tif"
        mask = tifffile.imread(os.path.join(input_dir, 'mask', fname)) > 0
        binw = apply_watershed(mask.astype(np.uint8))

        contours, _ = cv2.findContours(binw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Calcul des centroïdes: union contours + CSV
        coords = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                coords.append([M['m10']/M['m00'], M['m01']/M['m00']])
        # Ajouter centroïdes issus CSV manquants
        for _, r in df_tile.iterrows():
            pt = [r['Centroid_X'], r['Centroid_Y']]
            if not any(np.hypot(pt[0]-c[0], pt[1]-c[1])<1 for c in coords):
                coords.append(pt)
        coords = np.array(coords)
        if len(coords) < 2:
            continue

        # Arbre KD pour voisinage
        tree = KDTree(coords)
        # Determination angle principal
        angles_for_hist = []
        for i, pt in enumerate(coords):
            dists, idxs = tree.query(pt, k=13)
            angles_for_hist += [angle_between(pt, coords[j]) for j in idxs[1:]]
        hist, bins = np.histogram(angles_for_hist, bins=36, range=(0,180))
        main_ang = bins[np.argmax(hist)]

        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        # Ajout d'arêtes filtrées
        for i, pt in enumerate(coords):
            dists, idxs = tree.query(pt, k=13)
            for j, d in zip(idxs[1:], dists[1:]):
                ang = angle_between(pt, coords[j])
                if d < distance_threshold and abs((ang-main_ang+90)%180 -90) < tolerance_angle:
                    G.add_edge(i, j)

        # Recherche des chaînes alignées
        chains = []
        visited = set()
        for n in G.nodes:
            if n in visited:
                continue
            chain, queue = [n], [n]
            visited.add(n)
            while queue:
                u = queue.pop()
                for v in G.neighbors(u):
                    if v not in visited and is_aligned(coords[u], coords[v], main_ang):
                        visited.add(v); chain.append(v); queue.append(v)
            if len(chain) >= 3:
                chains.append(chain)
        if not chains:
            continue

        # Sélection score et chaîne optimale
        scores      = [score_file(c, coords, df_tile) for c in chains]
        best_idx    = int(np.argmax(scores))
        best_chain  = chains[best_idx]
        ordered     = sorted(best_chain, key=lambda i: coords[i][0])

        # Calcul épaisseur corrigée
        thicknesses = []
        for i in range(len(ordered)-1):
            p1, p2 = coords[ordered[i]], coords[ordered[i+1]]
            line = np.round(
                np.linspace(p1, p2, num=int(np.hypot(*(p2-p1))), endpoint=True)
            ).astype(int)
            h, w = binw.shape
            ys = np.clip(line[:, 1], 0, h - 1)
            xs = np.clip(line[:, 0], 0, w - 1)
            mask_vals = binw[ys, xs]
            full_d    = np.linalg.norm(p2-p1)
            inside_d  = np.count_nonzero(mask_vals)
            thicknesses.append(full_d - inside_d)
            
        # Visualisation de la chaîne sélectionnée
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(binw, cmap='gray')
        ax.set_title(f"Tile {tile} - Best Aligned Chain")

        # Dessin des centroïdes
        for i, (x, y) in enumerate(coords):
            ax.plot(x, y, 'o', color='blue', markersize=4)

        # Chaîne optimale en rouge
        for i in range(len(ordered) - 1):
            x0, y0 = coords[ordered[i]]
            x1, y1 = coords[ordered[i+1]]
            ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2)
            ax.plot(x0, y0, 'ro', markersize=5)
        # Dernier point
        x_last, y_last = coords[ordered[-1]]
        ax.plot(x_last, y_last, 'ro', markersize=5)

        # Sauvegarde de la figure
        plot_fname = os.path.join(plots_dir, f"{specimen}_{tile}_chain.png")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_fname, dpi=300)
        plt.close()

        # Sauvegarde résultats par cellule
        for order, idx in enumerate(ordered):
            x, y = coords[idx]
            cand = df_tile[(np.isclose(df_tile['Centroid_X'], x, atol=1))
                           & (np.isclose(df_tile['Centroid_Y'], y, atol=1))]
            if cand.empty:
                continue
            row = cand.iloc[0].copy()
            row['Corrected_CX']   = x
            row['Corrected_CY']   = y
            row['Tile_Num']       = tile_map[tile]
            row['File_ID']        = best_idx
            row['Chain_Order']    = order
            row['2p_Thickness']   = (thicknesses[order]
                                     if order < len(thicknesses) else np.nan)
            all_best.append(row)

    # Export CSV final
    df_out = pd.DataFrame(all_best)
    df_out.sort_values(['Tile_Num','File_ID','Chain_Order'], inplace=True)
    out_csv = os.path.join(csv_dir, f'results_{specimen}.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Fichier sauvegardé : {out_csv}")

print('Terminé')
