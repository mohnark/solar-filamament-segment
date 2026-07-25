import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
 
 
class SolarFilamentDataset(Dataset):
    def __init__(self, file_names, images_dir, masks_dir=None, transform=None, is_test=False):
        """
        file_names: list of base file names, e.g. "20260901165702Bh.jpeg"
        images_dir: directory containing the images
        masks_dir: directory containing precomputed masks (None if is_test=True)
        transform: an Albumentations transform, applied jointly to image and mask
        is_test: if True, only the image is loaded and returned (no mask)
        """
        self.file_names = file_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.is_test = is_test
 
        if not is_test and masks_dir is None:
            raise ValueError("masks_dir is required when is_test=False")
 
    def __len__(self):
        return len(self.file_names)
 
    def _load_image(self, file_name):
        """Load image as single-channel, kept as (H, W)."""
        image_path = os.path.join(self.images_dir, file_name)
        image = Image.open(image_path).convert("L")  # force grayscale, not RGB
        return np.array(image)
 
    def _load_mask(self, file_name):
        """Load precomputed mask, convert 0/255 to 0/1."""
        base_name = os.path.splitext(file_name)[0]
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # 0/255 -> 0/1
        return mask
 
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = self._load_image(file_name)
 
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image, file_name
 
        mask = self._load_mask(file_name)
 
        if self.transform:
            # same transform applied to both, so they stay aligned
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
 
        # mask stays float, single channel, shape (1, H, W) expected by loss functions
        mask = mask.unsqueeze(0).float()
 
        return image, mask
 