import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.spatial import KDTree
import networkx as nx
from utils import numerical_sort

# Paramètres pour recherche de la meilleure file
distance_threshold = 50  # distance max pour relier deux centroïdes
tolerance_angle = 10     # tolérance angulaire
score_area = 1           # poids pour variation d'aires
score_angle = 1          # poids pour variation d'angles

# Dossiers d'entrée et de sortie
input_dir = "/home/killian/sam2/inferences/15492/"
output_dir = "/home/killian/sam2/Results/"
os.makedirs(output_dir, exist_ok=True)

# Sous-dossiers plots et csv
plots_dir = os.path.join(output_dir, "Plots")
csv_dir = os.path.join(output_dir, "Data")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Nom du spécimen
specimen_name = os.path.basename(os.path.normpath(input_dir))
plots_dir = os.path.join(plots_dir, specimen_name)
csv_dir = os.path.join(csv_dir, specimen_name)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Fonctions utilitaires

def angle_between(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(delta[1], delta[0])) % 180


def is_aligned(p1, p2, ref_angle, tol=tolerance_angle):
    ang = angle_between(p1, p2)
    return abs((ang - ref_angle + 90) % 180 - 90) < tol


def score_file(chain, coords, df_tile):
    n_cells = len(chain)
    # aires
    areas = [df_tile[(np.isclose(df_tile['Centroid_X'], coords[i][0])) &
                     (np.isclose(df_tile['Centroid_Y'], coords[i][1]))]['Area'].values[0]
             for i in chain]
    area_std = np.std(areas)
    # angles
    angles = [angle_between(coords[chain[j-1]], coords[chain[j]]) for j in range(1, len(chain))]
    angle_std = np.std(angles) if len(angles) > 1 else 0
    return n_cells - score_area * area_std - score_angle * angle_std

# Traitement des CSV
csv_files = sorted(glob.glob(os.path.join(input_dir, "mask_measurements_*.csv")), key=numerical_sort)

for input_csv in csv_files:
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"Fichier vide : {input_csv}, ignoré.")
        continue
    has_tile = 'Tile_ID' in df.columns
    all_tiles = df['Tile_ID'].unique() if has_tile else [None]
    results_all = []

    specimen = os.path.splitext(os.path.basename(input_csv))[0].replace('mask_measurements_', '')

    for tile in all_tiles:
        df_tile = df[df['Tile_ID'] == tile] if has_tile else df.copy()
        # éviter doublons
        df_tile = df_tile.drop_duplicates(subset=['Centroid_X','Centroid_Y'])

        # charger masque
        tif_name = f"{specimen}_{tile}_mask.tif" if tile else f"{specimen}_mask.tif"
        mask = tifffile.imread(os.path.join(input_dir, 'mask', tif_name))
        binary = (mask>0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # coords
        coords = df_tile[['Centroid_X','Centroid_Y']].values
        if len(coords) < 3:
            continue
        tree = KDTree(coords)
        pairs = tree.query_pairs(distance_threshold)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        G.add_edges_from(pairs)
        # direction principale
        hist, bins = np.histogram([angle_between(coords[i], coords[j]) for i,j in G.edges], bins=36, range=(0,180))
        main_dir = bins[np.argmax(hist)]

        # extraire chains
        files = []
        for node in G.nodes:
            chain, queue, visited = [node],[node],{node}
            while queue:
                cur = queue.pop()
                for nei in G.neighbors(cur):
                    if nei not in visited and is_aligned(coords[cur],coords[nei],main_dir):
                        visited.add(nei); queue.append(nei); chain.append(nei)
            if len(chain)>=3:
                files.append(chain)
        if not files:
            continue
        scores = [score_file(ch,coords,df_tile) for ch in files]
        best_id = int(np.argmax(scores))
        best_chain = files[best_id]

        # extraire chaque cellule de la meilleure file
        for rank, idx in enumerate(best_chain,1):
            row = df_tile[(np.isclose(df_tile['Centroid_X'],coords[idx][0])) &
                          (np.isclose(df_tile['Centroid_Y'],coords[idx][1]))].iloc[0]
            results_all.append({
                'Tile_ID': tile,
                'file_id': best_id,
                'Rank_in_file': rank,
                'score_file': scores[best_id],
                'Mask_ID': row.get('Mask_ID', np.nan),
                'Area': row['Area'],
                'Equivalent_Diameter': row['Equivalent_Diameter'],
                'Centroid_X': row['Centroid_X'],
                'Centroid_Y': row['Centroid_Y']
            })

        # plot
        plt.figure(figsize=(6,6))
        plt.scatter(df_tile['Centroid_X'],df_tile['Centroid_Y'],color='gray',s=10)
        sel = pd.DataFrame([results_all[-len(best_chain)+i] for i in range(len(best_chain))])
        plt.scatter(sel['Centroid_X'],sel['Centroid_Y'],color='red',s=20)
        chain_pts = coords[best_chain]
        plt.plot(chain_pts[:,0],chain_pts[:,1],'-r')
        plt.gca().invert_yaxis()
        plt.title(f"Best file {tile}")
        plt.savefig(os.path.join(plots_dir,f"{specimen}_{tile}_bestfile.png"),dpi=300)
        plt.close()

    # sauvegarde CSV
    if results_all:
        df_out = pd.DataFrame(results_all)
        df_out.to_csv(os.path.join(csv_dir,os.path.basename(input_csv).replace('.csv','_bestfile.csv')),index=False)
        print("Sauvegardé:",os.path.basename(input_csv).replace('.csv','_bestfile.csv'))
