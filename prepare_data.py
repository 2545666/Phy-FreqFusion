"""
Data preparation utility.

1. Download UIEB / EUVP datasets (prints instructions).
2. Verify directory structure.
3. Generate train/test split index files.

Usage:
    python prepare_data.py --data_root ./data/UIEB --dataset uieb --verify
"""

import os
import sys
import argparse
import json
from pathlib import Path


UIEB_INFO = """
=== UIEB Dataset ===
Paper: "An Underwater Image Enhancement Benchmark Dataset and Beyond" (Li et al., TIP 2020)
URL:   https://li-chongyi.github.io/proj_benchmark.html

Download and organize as:
  {root}/
    raw/          # 890 degraded underwater images (.png)
    reference/    # 890 corresponding reference images (.png)

Files should have MATCHING filenames between raw/ and reference/.
Standard split: first 800 for training, last 90 for testing.
"""

EUVP_INFO = """
=== EUVP Dataset ===
Paper: "Fast Underwater Image Enhancement for Improved Visual Perception" (Islam et al., RAL 2020)
URL:   https://irvlab.cs.umn.edu/resources/euvp-dataset

Download the "underwater_scenes" subset and organize as:
  {root}/
    trainA/            # degraded training images
    trainB/            # reference training images
    validation/
      input/           # degraded validation images
      target/          # reference validation images
"""


def verify_uieb(root):
    """Verify UIEB dataset structure."""
    raw_dir = os.path.join(root, 'raw')
    ref_dir = os.path.join(root, 'reference')

    if not os.path.isdir(raw_dir):
        print(f"  [FAIL] Missing: {raw_dir}")
        return False
    if not os.path.isdir(ref_dir):
        print(f"  [FAIL] Missing: {ref_dir}")
        return False

    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    raw_files = {f.stem for f in Path(raw_dir).iterdir() if f.suffix.lower() in valid_ext}
    ref_files = {f.stem for f in Path(ref_dir).iterdir() if f.suffix.lower() in valid_ext}

    common = raw_files & ref_files
    raw_only = raw_files - ref_files
    ref_only = ref_files - raw_files

    print(f"  Raw images:       {len(raw_files)}")
    print(f"  Reference images: {len(ref_files)}")
    print(f"  Matched pairs:    {len(common)}")

    if raw_only:
        print(f"  [WARN] {len(raw_only)} raw images have no reference")
    if ref_only:
        print(f"  [WARN] {len(ref_only)} reference images have no raw counterpart")

    if len(common) >= 800:
        print(f"  [OK] Sufficient for 800/90 train/test split")
        return True
    else:
        print(f"  [WARN] Only {len(common)} pairs found (expected ~890)")
        return len(common) > 0


def verify_euvp(root):
    """Verify EUVP dataset structure."""
    required = ['trainA', 'trainB']
    optional = ['validation/input', 'validation/target']

    ok = True
    for d in required:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            n = len(list(Path(path).glob('*')))
            print(f"  {d}: {n} files")
        else:
            print(f"  [FAIL] Missing: {path}")
            ok = False

    for d in optional:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            n = len(list(Path(path).glob('*')))
            print(f"  {d}: {n} files")
        else:
            print(f"  [INFO] Missing (optional): {path}")

    return ok


def generate_split_file(root, output_path, n_train=800):
    """Generate JSON split file for reproducibility."""
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    raw_dir = os.path.join(root, 'raw')
    ref_dir = os.path.join(root, 'reference')

    raw_files = {f.stem: f.name for f in Path(raw_dir).iterdir()
                 if f.suffix.lower() in valid_ext}
    ref_files = {f.stem for f in Path(ref_dir).iterdir()
                 if f.suffix.lower() in valid_ext}

    common = sorted(set(raw_files.keys()) & ref_files)

    split = {
        'dataset': 'UIEB',
        'total': len(common),
        'train': common[:n_train],
        'test': common[n_train:],
        'n_train': min(n_train, len(common)),
        'n_test': max(0, len(common) - n_train),
    }

    with open(output_path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"  Split file saved: {output_path}")
    print(f"  Train: {split['n_train']}, Test: {split['n_test']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--dataset', type=str, default='uieb', choices=['uieb', 'euvp'])
    p.add_argument('--verify', action='store_true')
    p.add_argument('--generate_split', action='store_true')
    args = p.parse_args()

    if args.dataset == 'uieb':
        print(UIEB_INFO.format(root=args.data_root))
        if args.verify:
            print("Verifying UIEB dataset...")
            ok = verify_uieb(args.data_root)
            print("PASSED" if ok else "ISSUES FOUND")
        if args.generate_split:
            print("\nGenerating split file...")
            generate_split_file(args.data_root,
                                os.path.join(args.data_root, 'split.json'))
    else:
        print(EUVP_INFO.format(root=args.data_root))
        if args.verify:
            print("Verifying EUVP dataset...")
            ok = verify_euvp(args.data_root)
            print("PASSED" if ok else "ISSUES FOUND")


if __name__ == '__main__':
    main()
