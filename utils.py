import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def get_centroids_threshold(image):
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)

    return centroids


def plot_results(output_path, image_np, res_merge, centroids=None):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    if centroids.any():
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='+', linewidth=1.25)
    plt.title("Image Originale")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res_merge, cmap='gray')
    plt.title("Masques Prédits")
    plt.axis('off')
    
    plt.savefig(output_path)
    plt.close()