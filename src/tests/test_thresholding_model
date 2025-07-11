import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchmetrics
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob

# Initialisation de l'environnement
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Dataloader et Dataset
class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.img_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        self.mask_files = []
        valid_img_files = []
        for img_path in self.img_files:
            mask_path = os.path.splitext(img_path)[0] + '_mask.png'
            if os.path.exists(mask_path):
                self.mask_files.append(mask_path)
                valid_img_files.append(img_path)
        self.img_files = valid_img_files
        self.transforms = transforms

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_files[index]).convert("RGB"))
        label = np.array(Image.open(self.mask_files[index]).convert("RGB"))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.img_files)

# Fonction principale
def main():
    # Paramètres
    dataset_path = '/home/killian/Annotations/Annotations'
    output_dir = "/home/killian/sam2/Seuillage"
    os.makedirs(output_dir, exist_ok=True)

    # Métriques
    device = get_device()
    miou = torchmetrics.JaccardIndex(task='binary', num_classes=1).to(device)
    precision = torchmetrics.Precision(task='binary').to(device)
    recall = torchmetrics.Recall(task='binary').to(device)
    f1_score = torchmetrics.F1Score(task='binary').to(device)

    csv_results_path = os.path.join(output_dir, "seuillage_results.csv")
    csv_mask_path = os.path.join(output_dir, "mask_measurements.csv")

    with open(csv_results_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Image_ID", "IoU", "Precision", "Recall", "F1_Score", "Num_Masks", "Num_Masks_Label"])
    with open(csv_mask_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Image_ID", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    # Données
    dataset = DataLoaderSegmentation(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1)

    total_miou = total_precision = total_recall = total_f1 = total_images = 0

    for idx, (image, mask) in enumerate(dataloader):
        print(f"Image {idx+1}/{len(dataloader)}", end='\r')
        img_name = os.path.basename(dataset.img_files[idx])

        image_np = image.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        mask_bin = (mask_np[:, :, 2] > 0).astype(np.uint8)

        # Seuillage Otsu + morpho
        gray = np.mean(image_np, axis=2).astype(np.uint8)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed_bin = (processed > 0).astype(np.uint8)

        # Évaluation
        pred_tensor = torch.tensor(processed_bin, dtype=torch.bool, device=device)
        label_tensor = torch.tensor(mask_bin, dtype=torch.bool, device=device)

        img_miou = miou(pred_tensor, label_tensor).item()
        img_precision = precision(pred_tensor, label_tensor).item()
        img_recall = recall(pred_tensor, label_tensor).item()
        img_f1 = f1_score(pred_tensor, label_tensor).item()

        # Accumulation des métriques
        total_miou += img_miou
        total_precision += img_precision
        total_recall += img_recall
        total_f1 += img_f1
        total_images += 1

        # Sauvegarde CSV
        with open(csv_results_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                img_name, img_miou, img_precision, img_recall, img_f1,
                np.unique(processed_bin).size - 1, mask_bin.sum()
            ])

        # Analyse morphologique
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(processed_bin)
        with open(csv_mask_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(1, num_labels):
                x, y = centroids[i]
                area = stats[i, cv2.CC_STAT_AREA]
                equiv_diam = np.sqrt(4 * area / np.pi)
                writer.writerow([img_name, i, x, y, area, equiv_diam])

        # Visualisation
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image d'origine")
        plt.imshow(image_np.astype(np.uint8))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Prédiction")
        plt.imshow(processed_bin, cmap='gray')
        for x, y in centroids:
            plt.scatter(x, y, color='red', s=30)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Label")
        plt.imshow(mask_bin, cmap='gray')
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"Image_{idx}.png"))
        plt.close()

    # Résultats finaux
    if total_images > 0:
        print(f"\n--- Statistiques Globales ---")
        print(f"mIoU moyen: {total_miou / total_images:.4f}")
        print(f"Précision moyenne: {total_precision / total_images:.4f}")
        print(f"Rappel moyen: {total_recall / total_images:.4f}")
        print(f"F1-Score moyen: {total_f1 / total_images:.4f}")

if __name__ == "__main__":
    main()
