from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np
import torch

# On définit le DataLoader
class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, transforms=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        # print(self.img_files)
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, f'{img_path[:-4]}_mask.png'))
        
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))
        label = Image.open(mask_path)
        label = np.array(label.convert("RGB"))
        # if self.transforms:
        #     image = self.transforms(torch.from_numpy(image).float())
                 
        return image, torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)