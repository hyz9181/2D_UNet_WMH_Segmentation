import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class WMH_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_wmh.jpg"))
        print(img_path)
        print(mask_path)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        print(image.shape)
        print(mask.shape)
        return image, mask
