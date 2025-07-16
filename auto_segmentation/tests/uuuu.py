import os
import csv
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchmetrics

# Initialisation de l'environnement
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Dataloader et Dataset
class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.mask_files = [f'{img_path[:-4]}_mask.png' for img_path in self.img_files]
        self.transforms = transforms

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_files[index]).convert("RGB"))
        label = np.array(Image.open(self.mask_files[index]).convert("RGB"))

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.img_files)

# Utils
def merge_preds(preds):
    masks = torch.stack([torch.tensor(p['segmentation'], dtype=torch.bool) for p in preds])
    return masks.any(dim=0)

def save_visuals(image, prediction, label, out_path):
    if image.dim() == 3 and image.shape[0] <= 4:
        image = image.permute(1, 2, 0)
    image = image.numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(image.astype(np.uint8)); plt.title("Image d'origine"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(prediction.cpu().numpy(), cmap='gray'); plt.title("Prédiction"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(label.cpu().numpy(), cmap='gray'); plt.title("Label"); plt.axis('off')
    plt.savefig(out_path)
    plt.close()

def filter_by_area_distribution(components_info, z_thresh=5):
    """
    Filtre les objets dont l'aire est trop éloignée de la moyenne.
    """
    if not components_info:
        return []

    areas = np.array([obj[2] for obj in components_info])
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    filtered = [
        obj for obj in components_info
        if abs(obj[2] - mean_area) <= z_thresh * std_area
    ]
    return filtered

# Main
def main():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Params
    dataset_path = '/home/killian/Annotations/Annotations'
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
    config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
    output_dir = "/home/killian/sam2/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Metriques
    device = get_device()
    miou = torchmetrics.JaccardIndex(task='binary', num_classes=1).to(device)
    precision = torchmetrics.Precision(task='binary').to(device)
    recall = torchmetrics.Recall(task='binary').to(device)
    f1_score = torchmetrics.F1Score(task='binary').to(device)

    # CSV
    result_path = os.path.join(output_dir, "sam2_results.csv")
    masks_path = os.path.join(output_dir, "mask_measurements.csv")
    with open(result_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Image_ID", "IoU", "Precision", "Recall", "F1_Score", "Num_Masks", "Num_Masks_Label"])
    with open(masks_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Image_ID", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    # Data
    dataloader = DataLoader(DataLoaderSegmentation(dataset_path), batch_size=1)

    # Model
    sam2 = build_sam2(config_path, checkpoint, device=device, apply_postprocessing=False)
    model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=40,
        points_per_batch=30,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.80,
        stability_score_offset=0.8,
        crop_n_layers=4,
        box_nms_thresh=0.80,
        crop_n_points_downscale_factor=1.5,
        min_mask_region_area=85.0,
        use_m2m=True,
    )

    size_threshold = 5000
    results_writer = csv.writer(open(result_path, 'a', newline=''))
    masks_writer = csv.writer(open(masks_path, 'a', newline=''))

    total_iou = total_precision = total_recall = total_f1 = 0
    total_images = 0

    for i, (batch, mask) in enumerate(dataloader):
        print(f"Image {i+1}/{len(dataloader)}", end='\r')
        batch = batch.to(device)

        for idx, (image, label) in enumerate(zip(batch, mask)):
            pred = model.generate(image.cpu().numpy())
            if not pred:
                continue

            mask_stack = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
            filtered = mask_stack[mask_stack.sum(dim=(1, 2)) <= size_threshold]

            if filtered.nelement() == 0:
                continue

            merged = filtered.any(dim=0).to(device)

            label_np = np.asarray(label)
            target = torch.as_tensor(label_np[:, :, 2], dtype=torch.bool, device=device) if label_np.ndim == 3 else torch.as_tensor(label_np, dtype=torch.bool, device=device)

            iou = miou(merged, target).item()
            p = precision(merged, target).item()
            r = recall(merged, target).item()
            f1 = f1_score(merged, target).item()

            results_writer.writerow([f"Image_{i}_{idx}", iou, p, r, f1, filtered.shape[0], (label_np > 0).sum()])

            # Analyse des masques
            components_info = []
            for mask_id, m in enumerate(filtered):
                mask_np = m.cpu().numpy().astype(np.uint8)
                _, _, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
                for j in range(1, len(centroids)):
                    x, y = centroids[j]
                    area = stats[j, cv2.CC_STAT_AREA]
                    equiv_diam = np.sqrt(4 * area / np.pi)
                    components_info.append((x, y, area, equiv_diam, mask_id))

            # Filtrage basé sur la distribution des aires
            filtered_components = filter_by_area_distribution(components_info)

            for (x, y, area, equiv_diam, mask_id) in filtered_components:
                masks_writer.writerow([f"Image_{i}_{idx}", mask_id, x, y, area, equiv_diam])

            save_visuals(image.detach().cpu(), merged, target, os.path.join(output_dir, f"Image_{i}_{idx}.png"))

            total_iou += iou
            total_precision += p
            total_recall += r
            total_f1 += f1
            total_images += 1

    if total_images > 0:
        avg_iou = total_iou / total_images
        avg_p = total_precision / total_images
        avg_r = total_recall / total_images
        avg_f1 = total_f1 / total_images

        print("\n--- Résultats globaux (affichés uniquement dans la console) ---")
        print(f"Nombre d'images traitées : {total_images}")
        print(f"mIoU moyen     : {avg_iou:.4f}")
        print(f"Précision moy. : {avg_p:.4f}")
        print(f"Recall moyen   : {avg_r:.4f}")
        print(f"F1-Score moyen : {avg_f1:.4f}")

if __name__ == "__main__":
    main()
