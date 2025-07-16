import os
import glob
import numpy as np
import tifffile
import csv
import cv2
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# base_path = "/path/to/your/directory"  

# size_threshold = 10000 # Threshold for filtering out oversized objects
# mettre dans un parser 

def parse_arguments():
    """
    Analyzes command-line arguments to configure input and output paths.

    Uses argparse to allow the user to specify :
        - the input directory containing .tif images to be segmented
        - the output directory for saving results in Data directory and Plot directory .

    Returns:
    argparse.Namespace: An object containing the arguments `input` (str) and `output` (str).
        
    Example of command-line use:
    python script.py --input /path/to/your/folder --output /path/to/results
    """
    parser = argparse.ArgumentParser(description="Mask analysis and export of CSV and plot results")
    parser.add_argument('--base_path', '-i', type=str, required=True,
                        help='path to your input directory with .tif images')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory where results (CSV, masks, etc.) will be saved')
    parser.add_argument('--size_threshold', type=int, required=False, default=10_000,
                        help='hyperparameter for maximum size to segment cells in px')
    return parser.parse_args()
 
class TilingDataset(Dataset):
    """
    A custom dataset for extracting patches from a raster image (.tif).
    Only a central horizontal band is used, divided into tiles with an optional overlap.

    Attributes:
        input_raster (str): Path to input raster file (.tif).
        tile_size (int): Size (squared) of extracted tiles. (Default 640)
        stride (int): Horizontal offset between each tile. (Default 630)
        transforms (callable, optional): Transforms to be applied to each tile.
        raster_shape (tuple): Dimensions of original raster image.
        height (int): Image height (in px).
        width (int): Image width (in px).
        channels (int): Number of channels in image (default 3).
        num_tiles_x (int): Number of extracted horizontal tiles.
        num_tiles_y (int): Set to 1, as only a central strip is cut.
        center_y (int): Horizontal center position for extraction.
        tile_y_start (int): Horizontal start position for extracting center strip.
    """
    def __init__(self, input_raster, tile_size=640, stride=630, transforms=None):
        """
        Initializes dataset.

        Args:
            input_raster (str): Raster file path (.tif).
            tile_size (int): Tile size squared (default 640).
            stride (int): Horizontal offset between tiles. (Default 630).
            transforms (callable): Transforms to be applied to each tile (No transforms).
        """
        self.input_raster = input_raster
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms 

        with tifffile.TiffFile(input_raster) as tif:
            self.raster_shape = tif.pages[0].shape  
        
        if len(self.raster_shape) == 2:
            self.height, self.width = self.raster_shape
            self.channels = 1
        
        else:
            self.height, self.width, self.channels = self.raster_shape

        self.num_tiles_x = max(1, (self.width - self.tile_size) // self.stride + 1)
        self.num_tiles_y = 1  
        self.center_y = self.height // 2  
        self.tile_y_start = max(0, self.center_y - self.tile_size // 2)  

    def __len__(self):
        """
        return the total number of horizontale tile 

        Returns:
            int: Total number of tile
        """
        return self.num_tiles_x  

    def __getitem__(self, idx):
        tile_x = self.width - self.tile_size - (idx * self.stride)
        tile_y = self.tile_y_start  
        tile_x = max(0, tile_x) 

        with tifffile.TiffFile(self.input_raster) as tif:
            band = tif.pages[0].asarray()
            # if len(band.shape) == 2:
                # band = band[..., np.newaxis]
                # band = np.repeat(band[:, :, np.newaxis], 3, axis=2)
            band = band[tile_y:tile_y + self.tile_size, tile_x:tile_x + self.tile_size, :]

        if self.transforms:
            band = self.transforms(band)
            
        # Convert in 3 chanels
        if band.shape[2] == 4:  # if the image is on RGBA
            band = band[:, :, :3]  # keep olny the 3 first (RGB)

        return band

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

def segment_cells(args):
    
    args = parse_arguments()
    base_path = args.base_path

    # load all .tif files
    image_paths = sorted(glob.glob(os.path.join(base_path, "*.tif")))  # List of tif files
    if not image_paths:
        print(f"No images found {base_path}")
        exit()

    # Take parent's name of file
    base_folder = os.path.basename(base_path)  

    # Generate an output folder for this dataset
    output_dir = os.path.join(args.output, base_folder)
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # make a global .csv fro each images
    csv_masks_file = os.path.join(output_dir, f"mask_measurements_{base_folder}.csv")
    with open(csv_masks_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Name", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])

    # load SAM2
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    
    #Hyperparametres for SAM2
    model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=40,  
        points_per_batch=30,  
        pred_iou_thresh=0.65,  
        stability_score_thresh=0.80,  
        stability_score_offset=0.8,
        crop_n_layers=2,  
        box_nms_thresh=0.60,  
        crop_n_points_downscale_factor=1.5, 
        min_mask_region_area=25.0,  
        use_m2m=True, 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loop over all images in the folder
    for path in image_paths:
        image_name = os.path.basename(path)  
        print(f"Processing {image_name}...")

        dataset = TilingDataset(path, tile_size=640, stride=630)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, image in enumerate(dataloader):
            print(f"  Image {i+1}/{len(dataloader)} - Progress : {i/len(dataloader):.2%}")

            image_np = image.squeeze(0).cpu().numpy()
            pred = model.generate(image_np)
            res_tensor = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
            filtered_tensor = res_tensor[res_tensor.sum(dim=(1, 2)) <= args.size_threshold]
            res_merge = filtered_tensor.any(dim=0)

            components_info = []

            for mask_id, mask_pred in enumerate(filtered_tensor):
                mask_np = mask_pred.numpy().astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

                for j in range(1, num_labels):  # Skip background
                    x, y = centroids[j]
                    area = stats[j, cv2.CC_STAT_AREA]
                    equivalent_diameter = np.sqrt(4 * area / np.pi)
                    components_info.append((x, y, area, equivalent_diameter, mask_id))

            # Filtring by quantile
            masks_info = filter_by_area_distribution(components_info, z_thresh=5)

            # Sort masks from top (Y max) to bottom (Y min) and from right (X max) to left (X min)
            masks_info.sort(key=lambda m: (-m[1], -m[0]))  

            # Define the specific path for each image
            csv_image_file = os.path.join(output_dir, f"mask_measurements_{image_name}.csv")

            # Check if the file already exists (to avoid rewriting the header for each tile)
            file_exists = os.path.isfile(csv_image_file)
            
            # Save binary mask in mask/ folder
            mask_output_path = os.path.join(mask_dir, f"{image_name}_Image_{i}_mask.tif")
            tifffile.imwrite(mask_output_path, res_merge.cpu().numpy().astype(np.uint8) * 255)

            # Open in append mode (‘a’) to add new tiles
            with open(csv_image_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Write header only if file does not yet exis
                if not file_exists:
                    writer.writerow(["Tile_ID", "Mask_ID", "Centroid_X", "Centroid_Y", "Area", "Equivalent_Diameter"])
                
                unique_masks = {}
                for x, y, area, equivalent_diameter, mask_id in masks_info:
                    key = (round(x, 2), round(y, 2))  # On arrondit pour éviter les flottants très proches considérés différents
                    if key not in unique_masks or area > unique_masks[key][2]:  # index 2 = area
                        unique_masks[key] = (x, y, area, equivalent_diameter, mask_id)

                # Write filtered data to file
                for x, y, area, equivalent_diameter, mask_id in unique_masks.values():
                    writer.writerow([f"Image_{i}", mask_id, x, y, area, equivalent_diameter])
               
            # Creation of the figure to display the predicted image and masks
            try : 
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
            
                # Saving images
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image_np)
                plt.title("Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(res_merge.cpu().numpy(), cmap='gray')
                plt.title("Mask")
                plt.axis('off')

                output_img_path = os.path.join(output_dir, f"{image_name}_Image_{i}.png")
                plt.savefig(output_img_path)
                plt.close()
            except ModuleNotFoundError : 
                print("Matplotlib not found, unable to generate plots")

    return

if __name__ == "__main__":
    
    args = parse_arguments()
    segment_cells