import os
from functools import lru_cache

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler

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
        return_meta=False,
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
        return_meta: if True (and is_test=False), also return file_name, y, x
            alongside image/mask tiles, so tiles can be stitched back into
            full images for image-level validation metrics
        """
        self.images_dir = images_dir
        self.transform = transform
        self.is_test = is_test
        self.return_meta = return_meta
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

        # cache image+mask together per file, so the tile_size**2/overlap tiles
        # belonging to one file share a single JPEG decode + polygon RLE decode
        self._load_file = lru_cache(maxsize=mask_cache_size)(self._load_file_uncached)

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

    def _load_file_uncached(self, file_name):
        """Decode image (and mask, if labeled) for a file once, cached per file."""
        image = self._load_image(file_name)
        mask = None if self.is_test else self._build_mask_uncached(file_name)
        return image, mask

    def __getitem__(self, idx):
        file_name, y, x = self.index[idx]

        image, full_mask = self._load_file(file_name)
        image_tile = extract_tile(image, y, x, self.tile_size)

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image_tile)
                image_tile = augmented["image"]
            # y, x returned too, so predictions can be stitched back later
            return image_tile, file_name, y, x

        mask_tile = extract_tile(full_mask, y, x, self.tile_size)

        if self.transform:
            # same transform applied to both, so they stay aligned
            augmented = self.transform(image=image_tile, mask=mask_tile)
            image_tile = augmented["image"]
            mask_tile = augmented["mask"]

        # mask stays float, single channel, shape (1, H, W) expected by loss functions
        mask_tile = mask_tile.unsqueeze(0).float()

        if self.return_meta:
            return image_tile, mask_tile, file_name, y, x

        return image_tile, mask_tile


class FileGroupedSampler(Sampler):
    """
    Yields tile indices grouped by file (file order shuffled each epoch,
    tiles within a file shuffled too), instead of shuffling every tile
    globally. Keeps accesses to the same file clustered together so
    SolarFilamentDataset's per-file image/mask cache actually hits: each
    file gets decoded ~once per epoch instead of ~once per tile.
    Use with DataLoader(..., sampler=..., shuffle=False).
    """

    def __init__(self, dataset, seed=0):
        self.tiles_per_file = len(dataset.tile_coords)
        assert len(dataset.index) % self.tiles_per_file == 0
        self.num_files = len(dataset.index) // self.tiles_per_file
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        file_order = rng.permutation(self.num_files)
        for f in file_order:
            block = np.arange(f * self.tiles_per_file, (f + 1) * self.tiles_per_file)
            rng.shuffle(block)
            yield from block.tolist()

    def __len__(self):
        return self.tiles_per_file * self.num_files
