"""
Dataset loaders for UIEB and EUVP benchmarks.

Expected directory structures:
  UIEB/
    raw/           # 890 degraded images
    reference/     # 890 reference images (same filenames)
    
  EUVP/
    underwater_dark/  or  underwater_scenes/
      trainA/      # degraded
      trainB/      # reference
      validation/
        ...
"""

import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedUIEDataset(Dataset):
    """
    Generic paired dataset for UIE tasks.
    Loads (degraded, reference) image pairs.
    """
    def __init__(self, input_dir, target_dir, img_size=256,
                 augment=True, max_samples=None):
        super().__init__()
        self.img_size = img_size
        self.augment = augment

        # Collect matching filenames
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        input_files = {f.stem: f for f in Path(input_dir).iterdir()
                       if f.suffix.lower() in valid_ext}
        target_files = {f.stem: f for f in Path(target_dir).iterdir()
                        if f.suffix.lower() in valid_ext}

        common = sorted(set(input_files.keys()) & set(target_files.keys()))
        if max_samples:
            common = common[:max_samples]

        self.pairs = [(str(input_files[k]), str(target_files[k])) for k in common]
        print(f"[Dataset] Found {len(self.pairs)} paired images "
              f"from {input_dir}")

    def __len__(self):
        return len(self.pairs)

    def _sync_augment(self, img_in, img_gt):
        """Apply synchronized random augmentations."""
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(img_in, (self.img_size, self.img_size))
        img_in = TF.crop(img_in, i, j, h, w)
        img_gt = TF.crop(img_gt, i, j, h, w)

        # Random horizontal flip
        if random.random() > 0.5:
            img_in = TF.hflip(img_in)
            img_gt = TF.hflip(img_gt)

        # Random vertical flip
        if random.random() > 0.5:
            img_in = TF.vflip(img_in)
            img_gt = TF.vflip(img_gt)

        return img_in, img_gt

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        img_in = Image.open(inp_path).convert('RGB')
        img_gt = Image.open(gt_path).convert('RGB')

        # Resize to at least img_size (then crop if augmenting)
        min_side = self.img_size if not self.augment else self.img_size + 32
        img_in = TF.resize(img_in, min_side)
        img_gt = TF.resize(img_gt, min_side)

        if self.augment:
            img_in, img_gt = self._sync_augment(img_in, img_gt)
        else:
            img_in = TF.center_crop(img_in, self.img_size)
            img_gt = TF.center_crop(img_gt, self.img_size)

        img_in = TF.to_tensor(img_in)  # [0, 1]
        img_gt = TF.to_tensor(img_gt)

        return {
            'input': img_in,
            'target': img_gt,
            'filename': os.path.basename(inp_path),
        }


class UIEBDataset(PairedUIEDataset):
    """
    UIEB dataset (890 paired images).
    
    Args:
        root:  path to UIEB/ directory
        split: 'train' (first 800) or 'test' (last 90)
    """
    def __init__(self, root, split='train', img_size=256, augment=True):
        input_dir = os.path.join(root, 'raw')
        target_dir = os.path.join(root, 'reference')
        super().__init__(input_dir, target_dir, img_size, augment)

        # Standard split: 800 train / 90 test
        if split == 'train':
            self.pairs = self.pairs[:800]
        elif split == 'test':
            self.pairs = self.pairs[800:]
        print(f"[UIEB-{split}] Using {len(self.pairs)} images")


class EUVPDataset(PairedUIEDataset):
    """
    EUVP dataset (multiple scene subsets).
    
    Args:
        root:    path to EUVP/{scene}/ directory
        split:   'train' or 'val'
    """
    def __init__(self, root, split='train', img_size=256, augment=True):
        if split == 'train':
            input_dir = os.path.join(root, 'trainA')
            target_dir = os.path.join(root, 'trainB')
        else:
            input_dir = os.path.join(root, 'validation', 'input')
            target_dir = os.path.join(root, 'validation', 'target')
            augment = False
        super().__init__(input_dir, target_dir, img_size, augment)


class TestDataset(Dataset):
    """Single-image test dataset (no ground truth needed)."""
    def __init__(self, input_dir, img_size=256):
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.files = sorted([
            str(f) for f in Path(input_dir).iterdir()
            if f.suffix.lower() in valid_ext
        ])
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.img_size)
        img = TF.center_crop(img, self.img_size)
        img = TF.to_tensor(img)
        return {'input': img, 'filename': os.path.basename(path)}
