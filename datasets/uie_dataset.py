"""
Dataset loaders for UIE benchmarks.

Supported datasets:
  - UIEB  (890 paired images, standard 800/90 split)
  - EUVP  (trainA/trainB + validation splits)
  - LSUI  (large-scale paired dataset for generalization)
  - RUIE  (real-world unpaired dataset, no-reference eval only)

Expected directory structures:

  UIEB/
    raw/           # 890 degraded images
    reference/     # 890 reference images (same filenames)

  EUVP/
    underwater_dark/  or  underwater_scenes/
      trainA/      # degraded
      trainB/      # reference
      validation/
        input/
        target/

  LSUI/
    input/         # degraded underwater images
    GT/            # ground-truth reference images (same filenames)

  RUIE/
    UCCS/          # color cast subset     (no GT)
    UIQS/          # image quality subset  (no GT)
    UHTS/          # hazy/turbid subset    (no GT)
    # ---- OR a flat folder of images for simple evaluation ----
"""

import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# =============================================================================
#  Base Paired Dataset (unchanged logic)
# =============================================================================

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


# =============================================================================
#  UIEB Dataset  (UNCHANGED — preserves standard 800:90 split)
# =============================================================================

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


# =============================================================================
#  EUVP Dataset  (UNCHANGED)
# =============================================================================

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


# =============================================================================
#  NEW — LSUI Dataset  (paired, for full-reference generalization eval)
# =============================================================================

class LSUIDataset(PairedUIEDataset):
    """
    LSUI (Large-Scale Underwater Image) dataset.

    Expected layout:
        LSUI/
          input/    # degraded underwater images
          GT/       # ground-truth references  (matching filenames)

    Since this is used purely for generalization testing, augmentation
    defaults to False and no train/test split is applied — we evaluate
    on all available pairs.

    Args:
        root:       path to LSUI/ directory
        img_size:   evaluation resolution (default 256)
        max_samples: optional cap on number of images (useful for quick runs)
    """
    def __init__(self, root, img_size=256, max_samples=None):
        input_dir = os.path.join(root, 'input')
        target_dir = os.path.join(root, 'GT')

        # Fall back to alternative folder names that some LSUI mirrors use
        if not os.path.isdir(input_dir):
            input_dir = os.path.join(root, 'raw')
        if not os.path.isdir(target_dir):
            target_dir = os.path.join(root, 'reference')

        super().__init__(input_dir, target_dir, img_size,
                         augment=False, max_samples=max_samples)
        print(f"[LSUI] Loaded {len(self.pairs)} paired images for evaluation")


# =============================================================================
#  NEW — RUIE Dataset  (unpaired, no-reference evaluation only)
# =============================================================================

class RUIEDataset(Dataset):
    """
    RUIE (Real-world Underwater Image Enhancement) dataset.

    RUIE contains three challenge subsets with NO ground-truth references:
      - UCCS  (Underwater Color Cast Set)
      - UIQS  (Underwater Image Quality Set)
      - UHTS  (Underwater Hazy/Turbid Set)

    This loader can point at any single subset folder or the RUIE root
    (in which case it merges all subsets).

    Evaluation is limited to no-reference metrics (UCIQE, UIQM).

    Expected layouts (either works):
        RUIE/UCCS/  *.png|*.jpg ...
        RUIE/UIQS/  *.png|*.jpg ...
        RUIE/UHTS/  *.png|*.jpg ...
      OR
        RUIE/       *.png|*.jpg ...          (flat folder)

    Args:
        root:       path to RUIE/ or a specific subset folder
        subset:     None (auto-detect / all), 'UCCS', 'UIQS', or 'UHTS'
        img_size:   evaluation resolution
        max_samples: optional cap
    """
    VALID_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    def __init__(self, root, subset=None, img_size=256, max_samples=None):
        super().__init__()
        self.img_size = img_size
        self.files = []

        if subset is not None:
            # Load a specific subset
            subset_dir = os.path.join(root, subset)
            self.files = self._scan_dir(subset_dir)
        else:
            # Try known subset folders first
            for sub in ['UCCS', 'UIQS', 'UHTS']:
                sub_dir = os.path.join(root, sub)
                if os.path.isdir(sub_dir):
                    self.files.extend(self._scan_dir(sub_dir))
            # If none found, treat root as a flat image folder
            if not self.files:
                self.files = self._scan_dir(root)

        self.files.sort()
        if max_samples and len(self.files) > max_samples:
            self.files = self.files[:max_samples]

        print(f"[RUIE] Loaded {len(self.files)} images "
              f"(no-reference evaluation only)")

    def _scan_dir(self, dirpath):
        if not os.path.isdir(dirpath):
            return []
        return [str(f) for f in Path(dirpath).iterdir()
                if f.suffix.lower() in self.VALID_EXT]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.img_size)
        img = TF.center_crop(img, self.img_size)
        img = TF.to_tensor(img)
        return {
            'input': img,
            'filename': os.path.basename(path),
        }


# =============================================================================
#  TestDataset  (UNCHANGED — generic single-image inference loader)
# =============================================================================

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
