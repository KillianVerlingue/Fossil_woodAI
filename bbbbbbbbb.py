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
from itertools import combinations
from utils import numerical_sort

# Paramètres
distance_threshold = 45   # distance max pour relier deux centroïdes
tolerance_angle   = 12   # tolérance variation d'angles
Sblt              = 0.02 # sensibilité watershed
cells_per_tile    = 40   # normalisation nombre cellules

# Dossiers
tile_map = {}
tile_counter = 1
input_dir  = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
plots_dir  = os.path.join(output_dir, "Plots")
csv_dir    = os.path.join(output_dir, "Data")
specimen   = os.path.basename(os.path.normpath(input_dir))
plots_dir  = os.path.join(plots_dir, specimen)
csv_dir    = os.path.join(csv_dir, specimen)
for d in [output_dir, plots_dir, csv_dir]: os.makedirs(d, exist_ok=True)

# Utilitaires

def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180

# Fusion de chaînes proches et alignées
def merge_chains(chains, coords, max_gap=distance_threshold, tol_angle=tolerance_angle):
    merged = True
    while merged:
        merged = False
        for a, b in combinations(chains, 2):
            # endpoints
            a_end = coords[a[-1]]
            b_start = coords[b[0]]
            ang = angle_between(a_end, b_start)
            if (np.linalg.norm(a_end - b_start) < max_gap and
                abs((ang - angle_between(coords[a[0]], coords[a[-1]]) + 90) % 180 - 90) < tol_angle):
                # fusion
                a.extend(b)
                chains.remove(b)
                merged = True
                break
        if merged:
            continue
    return chains

# Scoring basé sur la longueur et rectitude
def score_chain(chain, coords):
    # longueur relative
    length_score = len(chain) / cells_per_tile
    # rectitude (RMS distance à la droite)
    xs = coords[chain,0]
    ys = coords[chain,1]
    a, b = np.polyfit(xs, ys, 1)
    dists = np.abs(a*xs - ys + b) / np.hypot(a, 1)
    rms = np.sqrt(np.mean(dists**2))
    straight_score = 1 - min(rms/20.0, 1.0)
    # score global (longueur prioritaire)
    return round(0.7 * length_score + 0.3 * straight_score, 4)

# Fonction watershed
def apply_watershed(mask):
    k = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, k, iterations=2)
    bg = cv2.dilate(opening, k, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, Sblt*dist.max(), 255, 0)
    fg = fg.astype(np.uint8)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown==255] = 0
    m = cv2.watershed(cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), markers)
    return (m>1).astype(np.uint8)

# Parcours des CSV
auto_csv = sorted(glob.glob(os.path.join(input_dir, 'mask_measurements_*.csv')), key=numerical_sort)
for csv_path in auto_csv:
    df_all = pd.read_csv(csv_path)
    if df_all.empty: continue
    spec = os.path.splitext(os.path.basename(csv_path))[0].split('_')[-1]
    tiles = sorted(df_all['Tile_ID'].unique(), key=numerical_sort)
    results = []
    for tile in tiles:
        if tile not in tile_map:
            tile_map[tile] = tile_counter
            tile_counter += 1
        num = tile_map[tile]

        df_tile = df_all[df_all['Tile_ID']==tile].drop_duplicates(['Tile_ID','Mask_ID'], keep='first')
        mask = tifffile.imread(os.path.join(input_dir,'mask',f"{spec}_{tile}_mask.tif"))>0
        binw = apply_watershed(mask.astype(np.uint8))
        ctrs,_ = cv2.findContours(binw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        coords = np.array([[cv2.moments(c)['m10']/cv2.moments(c)['m00'],
                             cv2.moments(c)['m01']/cv2.moments(c)['m00']] 
                            for c in ctrs if cv2.moments(c)['m00']>0])
        if len(coords)<3: continue
        # KD et graphe
        tree = KDTree(coords)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        # bâtir arêtes
        for i,p in enumerate(coords):
            d,i2 = tree.query(p, k=13)
            for j,dist in zip(i2[1:], d[1:]):
                ang = angle_between(p, coords[j])
                if dist<distance_threshold and abs((ang-angle_between(coords[0],coords[-1])+90)%180-90)<tolerance_angle:
                    G.add_edge(i,j)
        # extraire chaînes via composantes connexes ordonnées
        chains = [list(c) for c in nx.connected_components(G) if len(c)>=3]
        # fusionner chaînes potentielles
        chains = merge_chains(chains, coords)
        # scorer et récupérer meilleure
        scored = [(score_chain(c,coords),c) for c in chains]
        best = max(scored, key=lambda x:x[0])[1]

        # sauvegarde résultats
        for idx, node in enumerate(best):
            x,y = coords[node]
            rec = df_tile[(np.isclose(df_tile.Centroid_X,x,atol=1)) & (np.isclose(df_tile.Centroid_Y,y,atol=1))].iloc[0].to_dict()
            rec.update({'Tile_Num':num,'Chain_Order':idx})
            results.append(rec)

        # optionnel : visualisation rapide
        plt.figure(figsize=(4,4))
        plt.imshow(binw, cmap='gray')
        xs, ys = coords[best,0], coords[best,1]
        plt.plot(xs,ys,'-or')
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir,f"{spec}_{num}.png"),dpi=200)
        plt.close()

    # écriture CSV final
    df_out = pd.DataFrame(results).sort_values(['Tile_Num','Chain_Order'])
    df_out.to_csv(os.path.join(csv_dir,f"results_{spec}.csv"), index=False)
    print(f"Sauvegardé: results_{spec}.csv")
print('Terminé')
