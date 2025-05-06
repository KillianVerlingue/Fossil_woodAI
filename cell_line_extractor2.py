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

# Paramètres
distance_threshold = 50 # distance max pour relier deux centroïdes
tolerance_angle = 12     # tolérance variation d'angles
Sblt = 0.02              # sensibilité watershed
score_nb = 0.7           # poids nombre cellules
score_area = 0.1         # poids variation aires
score_angle = 0.2       # poids variation angles
cells_per_tile = 40     # normalisation nombre cellules

# Dossiers
# input_dir = "/home/killian/sam2/inferences/15485/"
input_dir = "/home/killian/sam2/inferences/15492/"
# input_dir = "/home/killian/sam2/inferences/11478/"
# input_dir = "/home/killian/sam2/inferences/13823/"
# input_dir = "/home/killian/sam2/inferences/TGV4/"
# input_dir = "/home/killian/sam2/inferences/TGV5/"
output_dir = "/home/killian/sam2/Results/"
plots_dir = os.path.join(output_dir, "Plots")
csv_dir = os.path.join(output_dir, "Data")
specimen_name = os.path.basename(os.path.normpath(input_dir))
plots_dir = os.path.join(plots_dir, specimen_name)
csv_dir = os.path.join(csv_dir, specimen_name)
for d in [output_dir, plots_dir, csv_dir]:
    os.makedirs(d, exist_ok=True)

# Fonctions

def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180

def is_aligned(p1, p2, ref_angle, tol=tolerance_angle):
    ang = angle_between(p1, p2)
    return abs((ang - ref_angle + 90) % 180 - 90) < tol

def score_file(chain, coords, df_tile):
    areas = []
    for idx in chain:
        x, y = coords[idx]
        row = df_tile[(np.isclose(df_tile['Centroid_X'], x, atol=1)) & (np.isclose(df_tile['Centroid_Y'], y, atol=1))]
        if not row.empty:
            areas.append(row.iloc[0]['Area'])

    if not areas or len(chain) < 2:
        return 0.0

    areas = np.array(areas)
    mean_area = areas.mean()
    area_std = areas.std() / mean_area if mean_area else 0
    area_score = 1 - min(area_std, 1.0)  

    angle_devs = [angle_between(coords[chain[i - 1]], coords[chain[i]]) for i in range(1, len(chain))]
    angle_var = np.std(angle_devs) / 180 if len(angle_devs) > 1 else 0
    angle_score = 1 - min(angle_var, 1.0)  

    length_score = min(len(chain) / cells_per_tile, 1.0)  

    # pondération
    score = (score_nb * length_score +
             score_area * area_score +
             score_angle * angle_score)

    return round(score, 4)  # toujours entre 0 et 1


def apply_watershed(binary_mask):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, Sblt * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR), markers)
    seg = np.zeros_like(binary_mask)
    seg[markers > 1] = 1
    return seg

# Lecture des CSV
csv_files = sorted(glob.glob(os.path.join(input_dir, 'mask_measurements_*.csv')), key=numerical_sort)

for csv_path in csv_files:
    all_best = []
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    specimen = os.path.splitext(os.path.basename(csv_path))[0].replace('mask_measurements_','')
    tiles = sorted(df['Tile_ID'].unique()) if 'Tile_ID' in df.columns else [None]

    for tile in tiles:
        df_tile = df[df['Tile_ID'] == tile].copy() if tile is not None else df.copy()
        if 'Tile_ID' in df.columns:
            df_tile = df_tile.sort_values('Area', ascending=False).drop_duplicates(subset=['Tile_ID','Mask_ID'])

        # Charger et segmenter le masque
        fname = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask = tifffile.imread(os.path.join(input_dir, 'mask', fname)) > 0
        binw = apply_watershed(mask.astype(np.uint8))
        binw = np.int32(binw)
        contours, _ = cv2.findContours(binw, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
        
        # Recalcul des centroïdes
        coords = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                coords.append([M['m10']/M['m00'], M['m01']/M['m00']])
        coords = np.array(coords)
        if len(coords) < 2:
            continue

        # Construction du graphe
        tree = KDTree(coords)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))

        # Détermination de l'angle principal (préliminaire)
        angles_for_hist = []
        k_for_hist = 12
        for i, pt in enumerate(coords):
            dists, idxs = tree.query(pt, k=k_for_hist+1)
            for j in idxs[1:]:
                angles_for_hist.append(angle_between(pt, coords[j]))
        hist_vals, bins = np.histogram(angles_for_hist, bins=36, range=(0,180))
        main_ang = bins[np.argmax(hist_vals)]

        # Construction avec filtre de distance et angle
        k = 12
        for i, pt in enumerate(coords):
            dists, idxs = tree.query(pt, k=k+1)
            for j, d in zip(idxs[1:], dists[1:]):
                if d < distance_threshold:
                    angle = angle_between(pt, coords[j])
                    if abs((angle - main_ang + 90) % 180 - 90) < tolerance_angle:
                        G.add_edge(i, j)

        # Détection des chaînes alignées
        chains = []
        vis = set()
        for n in G.nodes:
            if n in vis:
                continue
            chain, q = [n], [n]
            vis.add(n)
            while q:
                u = q.pop()
                for v in G.neighbors(u):
                    if v not in vis and is_aligned(coords[u], coords[v], main_ang):
                        vis.add(v)
                        chain.append(v)
                        q.append(v)
            if len(chain) >= 3:
                chains.append(chain)
        if not chains:
            continue


        # Sélection de la meilleure file
        scores = [score_file(c, coords, df_tile) for c in chains]
        best_idx = int(np.argmax(scores))
        best_chain = chains[best_idx]
        ordered = sorted(best_chain, key=lambda i: coords[i][0])

        # Calcul des 2p_Thickness
        corrected_thicknesses = []
        for i in range(len(ordered) - 1):
            p1 = coords[ordered[i]]
            p2 = coords[ordered[i + 1]]
            line = np.linspace(p1, p2, num=int(np.linalg.norm(p2 - p1)), endpoint=True)
            line = np.round(line).astype(int)
            mask_vals = binw[line[:, 1], line[:, 0]]
            full_dist = np.linalg.norm(p2 - p1)
            inside_dist = np.count_nonzero(mask_vals)
            corrected = full_dist - inside_dist
            corrected_thicknesses.append(corrected)

        # Collecte des données et visuels
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        axs[0].imshow(binw, cmap='gray'); axs[0].set_title('Masque Binaire'); axs[0].axis('off')
        axs[1].imshow(np.zeros_like(binw), cmap='gray');
        for cnt in contours:
            axs[1].plot(cnt[:,0,0], cnt[:,0,1], color='lime', linewidth=1)
        axs[1].scatter(coords[:,0], coords[:,1], color='red', s=10);
        axs[1].set_title('Centroïdes + Liens'); axs[1].axis('off')

        overlay = np.zeros((*binw.shape,3), dtype=np.uint8)
        norm = Normalize(vmin=min(scores), vmax=max(scores))
        cmap = colormaps['coolwarm']
        for idx_chain, chain in enumerate(chains):
            color = tuple(int(v) for v in (np.array(cmap(norm(scores[idx_chain]))[:3])*255)[::-1])
            for i in chain:
                cX, cY = map(int, coords[i])
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt,(cX,cY),False)>=0:
                        cv2.drawContours(overlay,[cnt],-1,color=color,thickness=cv2.FILLED)
                        break
        for i in ordered:
            cX,cY=map(int,coords[i])
            for cnt in contours:
                if cv2.pointPolygonTest(cnt,(cX,cY),False)>=0:
                    cv2.drawContours(overlay,[cnt],-1,color=(255,0,0),thickness=cv2.FILLED)
                    break
        axs[2].imshow(cv2.addWeighted(cv2.cvtColor(binw.astype(np.uint8)*255,cv2.COLOR_GRAY2RGB),0.3,overlay,0.7,0))
        axs[2].plot([coords[i][0] for i in ordered],[coords[i][1] for i in ordered],color='red',linewidth=2,marker='o')
        axs[2].set_title('Files Cellulaires Colorées'); axs[2].axis('off')
        plot_path = os.path.join(plots_dir, f"{specimen}_{tile}_plot.png")
        plt.tight_layout(); plt.savefig(plot_path, dpi=300); plt.close(fig)

        for order, i in enumerate(ordered):
            x,y = coords[i]
            cand = df_tile[(np.isclose(df_tile['Centroid_X'],x,atol=1)) & (np.isclose(df_tile['Centroid_Y'],y,atol=1))]
            if cand.empty:
                continue
            cand = cand.copy()
            cand['Corrected_CX'], cand['Corrected_CY'] = x, y
            cand['Tile_ID'], cand['File_ID'], cand['Order'] = tile, best_idx, order
            if order < len(corrected_thicknesses):
                cand['2p_Thickness'] = corrected_thicknesses[order]
            else:
                cand['2p_Thickness'] = np.nan
            all_best.append(cand.iloc[0].to_dict())

    # Sauvegarde du CSV pour chaque spécimen
    df_out = pd.DataFrame(all_best)
    df_out.sort_values(['Tile_ID','File_ID','Order'], inplace=True)
    output_csv = os.path.join(csv_dir, f'results_{specimen}.csv')
    df_out.to_csv(output_csv, index=False)
    print(f'Fichier sauvegardé : {output_csv}')

print('Terminé')
