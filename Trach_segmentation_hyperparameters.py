# Trach_segmentation_hyperparameters.py

def main():
    # Imports et initialisations
    import os
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from torch.utils.data import Dataset
    import glob
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    import torchmetrics
    import matplotlib.pyplot as plt
    from itertools import product

    # --- Fonction de filtrage par taille ---
    def filter_objects_by_size(tensor, threshold):
        """
        Filtre les objets (chaque channel) d'un tenseur de forme (C, W, H)
        en ne conservant que ceux dont le nombre de pixels à 1 est inférieur ou égal au seuil.
        
        Args:
            tensor (torch.Tensor): Tenseur de forme (C, W, H) contenant des masques booléens.
            threshold (int): Seuil maximal (en nombre de pixels à 1) autorisé.
        
        Returns:
            torch.Tensor: Tenseur filtré (nombre de channels potentiellement réduit).
        """
        # Calcul du nombre de pixels à 1 par channel
        ones_per_channel = tensor.sum(dim=(1, 2))
        # Conserver uniquement les channels dont le nombre de pixels à 1 est inférieur ou égal au seuil
        mask = ones_per_channel <= threshold
        return tensor[mask]

    # --- Chargement du modèle ---
    checkpoint = "/home/killian/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)
    
    # Création du générateur de masques (sera réinstancié avec différents hyperparamètres)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # --- Définition du DataLoader ---
    class DataLoaderSegmentation(Dataset):
        def __init__(self, folder_path, transforms=None):
            super(DataLoaderSegmentation, self).__init__()
            self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                base = os.path.splitext(img_path)[0]
                self.mask_files.append(f"{base}_mask.png")
            self.transforms = transforms

        def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            label = Image.open(mask_path).convert("RGB")
            label = np.array(label)
            if self.transforms:
                image = self.transforms(torch.from_numpy(image).float())
            return image, torch.from_numpy(label).float()

        def __len__(self):
            return len(self.img_files)

    # Création du dataset et DataLoader (batch_size=1 pour simplifier)
    dataset = DataLoaderSegmentation('/home/killian/Annotations/Annotations')
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Boucle d'expérimentation sur les hyperparamètres ---
    device = "cuda"

    hyperparams_grid = {
        'points_per_side': [20, 50],
        'points_per_batch': [20, 50],
        'pred_iou_thresh': [0.5, 0.9],
        'stability_score_thresh': [0.5, 1.0],
        'stability_score_offset': [0.5, 1.0],
        'crop_n_layers': [1, 2],
        'box_nms_thresh': [0.5, 1.0],
        'crop_n_points_downscale_factor': [2, 3],
        'min_mask_region_area': [10.0, 20.0],
        'use_m2m': [True, False]
    }

    # Dossier de sauvegarde des prédictions
    save_dir_base = "/home/killian/sam2/predictions"
    os.makedirs(save_dir_base, exist_ok=True)

    best_miou = -1.0
    best_params = None

    keys = list(hyperparams_grid.keys())
    combinations = list(product(*(hyperparams_grid[k] for k in keys)))
    print(f"Total configurations à tester : {len(combinations)}")

    # Valeur seuil pour filtrer les objets (en nombre de pixels à 1)
    size_threshold = 10000  # À ajuster 

    for combo in combinations:
        params = dict(zip(keys, combo))
        print("\n=== Test de la configuration ===")
        print(params)
        
        # Instanciation du générateur avec les hyperparamètres actuels
        model = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=params['points_per_side'],
            points_per_batch=params['points_per_batch'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            stability_score_offset=params['stability_score_offset'],
            crop_n_layers=params['crop_n_layers'],
            box_nms_thresh=params['box_nms_thresh'],
            crop_n_points_downscale_factor=params['crop_n_points_downscale_factor'],
            min_mask_region_area=params['min_mask_region_area'],
            use_m2m=params['use_m2m'],
        )
        
        miou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=1).to(device)
        total_miou = 0.0
        total_images = 0

        config_name = "_".join(f"{k}-{v}" for k, v in params.items())
        config_save_dir = os.path.join(save_dir_base, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        for i, (batch, mask) in enumerate(test_dataloader):
            print(f"Batch {i+1}/{len(test_dataloader)}")
            batch = batch.to(device)
            
            for j, (image, label) in enumerate(zip(batch, mask)):
                image_np = image.cpu().numpy()
                pred = model.generate(image_np.squeeze())
                if len(pred) == 0:
                    print("Aucun masque généré pour cette image.")
                    continue

                # Fusionner les masques générés en conservant chaque masque dans un channel
                res_tensor = torch.stack([torch.tensor(m['segmentation'], dtype=torch.bool) for m in pred])
                # Filtrer les objets trop grands
                filtered_tensor = filter_objects_by_size(res_tensor, size_threshold)
                if filtered_tensor.shape[0] == 0:
                    print("Aucun objet ne respecte le seuil de taille pour cette image.")
                    continue
                # Recombiner les masques filtrés
                res_merge = filtered_tensor.any(dim=0).to(device)
                
                trach = torch.as_tensor(label[:, :, 2], dtype=torch.bool, device=device)

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.title("Prédiction")
                plt.imshow(res_merge.cpu().numpy(), cmap='gray')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title("Label")
                plt.imshow(trach.cpu().numpy(), cmap='gray')
                plt.axis('off')
                
                file_name = os.path.join(config_save_dir, f"pred_batch{i+1}_img{j+1}.png")
                plt.savefig(file_name, bbox_inches='tight')
                plt.close()
                print(f"Figure sauvegardée : {file_name}")
                
                img_miou = miou_metric(res_merge, trach)
                total_miou += img_miou.item()
                total_images += 1

        if total_images > 0:
            mean_miou = total_miou / total_images
            print(f"mIoU moyen pour cette configuration : {mean_miou:.4f}")
        else:
            mean_miou = 0.0
            print("Aucune image traitée pour cette configuration.")
        
        if mean_miou > best_miou:
            best_miou = mean_miou
            best_params = params

    print("\n=== Résultat final ===")
    print(f"Meilleure configuration : {best_params}")
    print(f"mIoU moyen le plus élevé : {best_miou:.4f}")


if __name__ == "__main__":
    main()
