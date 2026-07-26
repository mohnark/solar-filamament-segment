import os
from functools import lru_cache

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .tiling import get_tile_coordinates, extract_tile
from .decode import load_annotations, build_file_to_annotations, build_combined_mask


class SolarFilamentDataset(Dataset):
    def __init__(
        self,
        file_names,
        images_dir,
        annotations_json=None,
        transform=None,
        is_test=False,
        tile_size=512,
        overlap=64,
        image_height=2048,
        image_width=2048,
        mask_cache_size=16,
    ):
        """
        file_names: list of base file names, e.g. "20260901165702Bh.jpeg"
        images_dir: directory containing the raw grayscale images
        annotations_json: path to the COCO-style annotation file (None if is_test=True).
            Masks are decoded from polygon segmentations on the fly, per tile,
            instead of being read from precomputed mask PNGs.
        transform: an Albumentations transform, applied jointly to image and mask tile
        is_test: if True, tiles are still built (for full-image inference later),
                 but no mask is loaded or returned
        tile_size, overlap: passed to tiling.get_tile_coordinates
        image_height, image_width: expected full image dimensions (2048x2048 here)
        mask_cache_size: number of full-image masks to keep decoded in memory,
            so the 512x512 tiles of the same image don't each re-run RLE decode
        """
        self.images_dir = images_dir
        self.transform = transform
        self.is_test = is_test
        self.tile_size = tile_size
        self.image_height = image_height
        self.image_width = image_width

        if not is_test:
            if annotations_json is None:
                raise ValueError("annotations_json is required when is_test=False")
            data = load_annotations(annotations_json)
            self.file_to_anns = build_file_to_annotations(data)
        else:
            self.file_to_anns = None

        # precompute tile coordinates once, shared across all images
        # (safe since all images are the same fixed size)
        self.tile_coords = get_tile_coordinates(image_height, image_width, tile_size, overlap)

        # build a flat index: one entry per (file_name, tile_coord) pair
        self.index = []
        for file_name in file_names:
            for (y, x) in self.tile_coords:
                self.index.append((file_name, y, x))

        self._build_mask = lru_cache(maxsize=mask_cache_size)(self._build_mask_uncached)

    def __len__(self):
        return len(self.index)

    def _load_image(self, file_name):
        """Load image as single-channel grayscale, kept as (H, W)."""
        image_path = os.path.join(self.images_dir, file_name)
        image = Image.open(image_path).convert("L")  # force grayscale, not RGB
        return np.array(image)

    def _build_mask_uncached(self, file_name):
        """Decode the full-image binary mask from polygon annotations."""
        anns = self.file_to_anns.get(file_name, [])
        if not anns:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        return build_combined_mask(anns, self.image_height, self.image_width)

    def __getitem__(self, idx):
        file_name, y, x = self.index[idx]

        image = self._load_image(file_name)
        image_tile = extract_tile(image, y, x, self.tile_size)

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image_tile)
                image_tile = augmented["image"]
            # y, x returned too, so predictions can be stitched back later
            return image_tile, file_name, y, x

        full_mask = self._build_mask(file_name)
        mask_tile = extract_tile(full_mask, y, x, self.tile_size)

        if self.transform:
            # same transform applied to both, so they stay aligned
            augmented = self.transform(image=image_tile, mask=mask_tile)
            image_tile = augmented["image"]
            mask_tile = augmented["mask"]

        # mask stays float, single channel, shape (1, H, W) expected by loss functions
        mask_tile = mask_tile.unsqueeze(0).float()

        return image_tile, mask_tile
