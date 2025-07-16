
import os
import csv
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchmetrics

size_threshold = 1280 #e de segmentation max

def get_device():
    """
    Returns the appropriate device for computation.

    Returns:
        str: 'cuda' if a GPU is available, else 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

# Dataloader et Dataset
class TilingDataset(Dataset):
    """
    A custom PyTorch Dataset for extracting tiles from high-resolution .tif raster images
    and their corresponding segmentation masks.

    Assumes that each .tif image has an associated binary mask with the same filename,
    but ending in `_mask.png`.

    Only a single tile is extracted per image, centered vertically.

    Attributes:
        image_paths (list): List of paths to .tif images.
        label_paths (list): List of paths to corresponding mask images.
        tile_size (int): Width and height of each square tile.
        stride (int): Horizontal stride between tiles (not used in this single-tile version).
        transforms (callable, optional): Optional transform to be applied to image tiles.
    """
    def __init__(self, input_folder, tile_size=640, stride=630, transforms=None):
        """
        Args:
            input_folder (str): Directory containing .tif images and corresponding _mask.png labels.
            tile_size (int, optional): Size of each square tile (default: 640).
            stride (int, optional): Horizontal stride for tiling (default: 630, currently unused).
            transforms (callable, optional): Optional transform to apply to image tiles.
        """
        self.image_paths = sorted(glob.glob(os.path.join(input_folder, '*.tif')))
        self.label_paths = [p.replace('.tif', '_mask.png') for p in self.image_paths]
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms

    def __len__(self):
        """
        Returns:
            int: Total number of image-mask pairs (one tile per image).
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Extracts a central tile from the input image and the corresponding tile from the mask.

        Args:
            idx (int): Index of the image-mask pair.

        Returns:
            tuple: (image_tile, label_tile)
                - image_tile (ndarray or tensor): Extracted RGB tile from the input image.
                - label_tile (ndarray): Corresponding grayscale tile from the binary mask.
        """
        # Load raster image
        with tifffile.TiffFile(self.image_paths[idx]) as tif:
            raster = tif.pages[0].asarray()

        if raster.ndim == 2:  # Grayscale
            raster = np.repeat(raster[:, :, np.newaxis], 3, axis=2)
        elif raster.shape[2] == 4:  # RGBA
            raster = raster[:, :, :3]  # Keep RGB channels

        height, width, _ = raster.shape
        center_y = height // 2
        tile_y = max(0, center_y - self.tile_size // 2)
        tile_x = max(0, width - self.tile_size - (0 * self.stride))  # Only one tile per image

        image_tile = raster[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        # Load corresponding label
        label = np.array(Image.open(self.label_paths[idx]).convert("L"))
        label_tile = label[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size]

        if self.transforms:
            image_tile = self.transforms(image_tile)

        return image_tile, label_tile

# Utils
def merge_preds(preds):
    """
    Merges a list of predicted masks into a single binary mask.

    Args:
        preds (list): List of predicted masks from SAM2.

    Returns:
        torch.Tensor: A single binary mask (H, W).
    """
    masks = torch.stack([torch.tensor(p['segmentation'], dtype=torch.bool) for p in preds])
    return masks.any(dim=0)

def save_visuals(image, prediction, label, out_path):
    """
    Saves a visual comparison of input image, prediction, and ground truth label.

    Args:
        image (torch.Tensor): Input image tensor (H, W, C).
        prediction (torch.Tensor): Predicted mask (H, W).
        label (torch.Tensor): Ground truth mask (H, W).
        out_path (str): Output path to save the visualization image.
    """
    if image.dim() == 3 and image.shape[0] <= 4:
        image = image.permute(1, 2, 0)
    image = image.numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(image.astype(np.uint8)); plt.title("Image"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(prediction.cpu().numpy(), cmap='gray'); plt.title("Segmentation"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(label.cpu().numpy(), cmap='gray'); plt.title("Label(Ground truth)"); plt.axis('off')
    plt.savefig(out_path)
    plt.close()

def filter_by_area_distribution(components_info, z_thresh=2.5):
    """
    Removes objects whose surface area is at variance with the area distribution.

    This function calculates the mean and standard deviation of object areas, then filters
    those whose area is more than `z_thresh` standard deviations from the mean.

    Args:
        components_info (list of tuple): List of tuples (x, y, area, equivalent_diameter, mask_id).
        z_thresh (float): z-score threshold for filtering (default 2.5).

    Returns:
        list: Filtered list of components containing only valid objects.
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

#Import des chemins et du modèle
def sam2_evaluate():
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
    dataset = TilingDataset(dataset_path, tile_size=640, stride=630)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model et hyperparamètres
    sam2 = build_sam2(config_path, checkpoint, device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=40,
        points_per_batch=30,
        pred_iou_thresh=0.50,
        stability_score_thresh=0.65,
        stability_score_offset=0.8,
        crop_n_layers=6,
        box_nms_thresh=0.60,
        crop_n_points_downscale_factor=1.5,
        min_mask_region_area=8.0,
        use_m2m=True
    )

    # Evaluation
    results_writer = csv.writer(open(result_path, 'a', newline=''))
    masks_writer = csv.writer(open(masks_path, 'a', newline=''))

    total_iou = total_precision = total_recall = total_f1 = 0 #on initie les metrics totales
    total_images = 0
    
    for i, (batch, mask) in enumerate(dataloader):
        print(f"Image {i+1}/{len(dataloader)}", end='\r')
        batch = batch.to(device)
        
        components_info = []

        for idx, (image, label) in enumerate(zip(batch, mask)):
            pred = generator.generate(image.cpu().numpy())
            if not pred:
                continue
            
            mask_stack = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
            filtered = mask_stack[mask_stack.sum(dim=(1, 2)) <= size_threshold]
            merged = filtered.any(dim=0).to(device)
            for mask_id, m in enumerate(filtered):
                mask_np = m.cpu().numpy().astype(np.uint8)
                _, _, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
               
                for j in range(1, len(centroids)):
                    x, y = centroids[j]
                    area = stats[j, cv2.CC_STAT_AREA]
                    equiv_diam = np.sqrt(4 * area / np.pi)
                    components_info.append((x, y, area, mask_id))
                    
                    masks_writer.writerow([f"Image_{i}_{idx}", mask_id, x, y, area, equiv_diam])
           
            label_np = np.asarray(label)
            target = torch.as_tensor(label_np[:, :, 2], dtype=torch.bool, device=device) if label_np.ndim == 3 else torch.as_tensor(label_np, dtype=torch.bool, device=device)
           
            valid_components = filter_by_area_distribution(components_info, z_thresh=5)
            valid_mask_ids = [obj[3] for obj in valid_components]
            filtered_mask_stack = torch.stack([filtered[i] for i in valid_mask_ids]) if valid_mask_ids else torch.zeros_like(filtered[0]).unsqueeze(0)
            
            if filtered.shape[0] == 0:
                merged = torch.zeros_like(image[0], dtype=torch.bool).to(device)
            else:
                merged = filtered.any(dim=0).to(device)


            iou = miou(merged, target).item()
            p = precision(merged, target).item()
            r = recall(merged, target).item()
            f1 = f1_score(merged, target).item()

            # On cumule les métriques
            total_iou += iou
            total_precision += p
            total_recall += r
            total_f1 += f1
            total_images += 1

            results_writer.writerow([f"Image_{i}_{idx}", iou, p, r, f1, filtered.shape[0], (label_np > 0).sum()])

            save_visuals(image.detach().cpu(), merged, target, os.path.join(output_dir, f"Image_{i}_{idx}.png"))

    # Affichage final
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
    sam2_evaluate