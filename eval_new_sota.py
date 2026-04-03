"""
eval_new_sota.py — Evaluate new SOTA baselines from pre-generated images.

Computes full-reference (PSNR, SSIM) and no-reference (UCIQE, UIQM)
metrics on enhanced images produced by external methods like PUGAN, SUT,
or any other baseline.

This script does NOT run any model — it reads folders of already-enhanced
images and compares them against ground-truth references.

Usage:
  # Evaluate a single method
  python eval_new_sota.py \
      --method PUGAN \
      --pred_dir ./results/PUGAN/uieb_test \
      --gt_dir ./data/UIEB/reference \
      --gt_index_dir ./data/UIEB/raw

  # Evaluate multiple methods at once
  python eval_new_sota.py \
      --methods PUGAN SUT WaterNet Ours \
      --pred_dirs ./results/PUGAN ./results/SUT ./results/WaterNet ./results/Ours \
      --gt_dir ./data/UIEB/reference \
      --gt_index_dir ./data/UIEB/raw \
      --output_csv comparison_results.csv \
      --latex

Notes on filename matching:
  The script matches prediction filenames to GT filenames by stem
  (ignoring extensions). If your predictions use different naming than
  UIEB, use --gt_index_dir to specify which 90 test images to use.
  The standard UIEB test set is the last 90 images when sorted
  alphabetically.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import calc_psnr, calc_ssim, calc_uciqe, calc_uiqm


# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate SOTA methods from pre-generated images')

    # --- Single-method mode ---
    p.add_argument('--method', type=str, default=None,
                   help='Name of a single method (e.g., PUGAN)')
    p.add_argument('--pred_dir', type=str, default=None,
                   help='Directory of enhanced images for --method')

    # --- Multi-method mode ---
    p.add_argument('--methods', type=str, nargs='+', default=None,
                   help='Names of multiple methods')
    p.add_argument('--pred_dirs', type=str, nargs='+', default=None,
                   help='Directories for each method (same order as --methods)')

    # --- Ground truth ---
    p.add_argument('--gt_dir', type=str, required=True,
                   help='Directory of ground-truth reference images')
    p.add_argument('--gt_index_dir', type=str, default=None,
                   help='If provided, use the last 90 sorted filenames from '
                        'this dir to select the test split from gt_dir. '
                        'Useful for UIEB where GT and raw share the same stems.')
    p.add_argument('--test_split_start', type=int, default=800,
                   help='Index to split test images (default 800 for UIEB)')

    # --- Options ---
    p.add_argument('--img_size', type=int, default=None,
                   help='Resize to this before metric computation. '
                        'None = use original resolution')
    p.add_argument('--output_csv', type=str, default=None,
                   help='Save all results to CSV')
    p.add_argument('--output_json', type=str, default=None,
                   help='Save per-image results to JSON')
    p.add_argument('--latex', action='store_true',
                   help='Print a LaTeX-formatted results table')

    return p.parse_args()


# =========================================================================
#  Helpers
# =========================================================================

VALID_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def load_image_as_tensor(path, img_size=None):
    """Load an image and convert to (C, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    if img_size is not None:
        img = TF.resize(img, img_size)
        img = TF.center_crop(img, img_size)
    return TF.to_tensor(img)


def get_test_stems(gt_index_dir, split_start=800):
    """
    Get the stems (filenames without extension) of the UIEB test split.
    Returns the last (total - split_start) filenames when sorted.
    """
    all_files = sorted([
        f.stem for f in Path(gt_index_dir).iterdir()
        if f.suffix.lower() in VALID_EXT
    ])
    return set(all_files[split_start:])


def collect_gt_files(gt_dir, test_stems=None):
    """
    Build a dict: {stem: filepath} for ground-truth images.
    If test_stems is provided, only include those stems.
    """
    gt_map = {}
    for f in Path(gt_dir).iterdir():
        if f.suffix.lower() in VALID_EXT:
            if test_stems is None or f.stem in test_stems:
                gt_map[f.stem] = str(f)
    return gt_map


def collect_pred_files(pred_dir):
    """Build a dict: {stem: filepath} for prediction images."""
    pred_map = {}
    for f in Path(pred_dir).iterdir():
        if f.suffix.lower() in VALID_EXT:
            pred_map[f.stem] = str(f)
    return pred_map


# =========================================================================
#  Per-method evaluation
# =========================================================================

def evaluate_method(method_name, pred_dir, gt_map, img_size=None):
    """
    Compute metrics for a single method.

    Returns:
        avg_metrics: dict with PSNR, SSIM, UCIQE, UIQM averages
        per_image:   list of per-image metric dicts
    """
    pred_map = collect_pred_files(pred_dir)

    # Match by stem
    common_stems = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common_stems:
        print(f"  [WARNING] No matching filenames between {pred_dir} "
              f"and GT directory! Found {len(pred_map)} predictions, "
              f"{len(gt_map)} GT images.")
        return {}, []

    unmatched = set(gt_map.keys()) - set(pred_map.keys())
    if unmatched:
        print(f"  [INFO] {len(unmatched)} GT images have no matching "
              f"prediction (expected if method processed a subset)")

    psnr_vals, ssim_vals, uciqe_vals, uiqm_vals = [], [], [], []
    per_image = []

    for stem in common_stems:
        pred_t = load_image_as_tensor(pred_map[stem], img_size)
        gt_t = load_image_as_tensor(gt_map[stem], img_size)

        p = calc_psnr(pred_t, gt_t)
        s = calc_ssim(pred_t.unsqueeze(0), gt_t.unsqueeze(0))
        u = calc_uciqe(pred_t)
        q = calc_uiqm(pred_t)

        psnr_vals.append(p)
        ssim_vals.append(s)
        uciqe_vals.append(u)
        uiqm_vals.append(q)

        per_image.append({
            'filename': stem,
            'PSNR': p, 'SSIM': s, 'UCIQE': u, 'UIQM': q,
        })

    avg = {
        'PSNR': float(np.mean(psnr_vals)),
        'SSIM': float(np.mean(ssim_vals)),
        'UCIQE': float(np.mean(uciqe_vals)),
        'UIQM': float(np.mean(uiqm_vals)),
    }

    print(f"  {method_name:20s} | "
          f"PSNR={avg['PSNR']:6.2f} | SSIM={avg['SSIM']:.4f} | "
          f"UCIQE={avg['UCIQE']:.4f} | UIQM={avg['UIQM']:.4f} | "
          f"n={len(common_stems)}")

    return avg, per_image


# =========================================================================
#  LaTeX table generation
# =========================================================================

def print_latex_table(all_results):
    """
    Print a publication-ready LaTeX table.

    Args:
        all_results: OrderedDict {method_name: {PSNR, SSIM, UCIQE, UIQM}}
    """
    metrics = ['PSNR', 'SSIM', 'UCIQE', 'UIQM']
    methods = list(all_results.keys())

    # Find best and second-best per metric
    best, second = {}, {}
    for m in metrics:
        ranked = sorted(methods,
                        key=lambda name: all_results[name].get(m, 0),
                        reverse=True)
        best[m] = ranked[0]
        second[m] = ranked[1] if len(ranked) > 1 else None

    print("\n% ---- Auto-generated LaTeX table ----")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Quantitative comparison on the UIEB test set (90 images). "
          r"\textbf{Bold}: best, \underline{underline}: second best.}")
    print(r"\label{tab:sota_comparison}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & "
          r"UCIQE$\uparrow$ & UIQM$\uparrow$ \\")
    print(r"\midrule")

    for name in methods:
        row = [name.replace('_', r'\_')]
        for m in metrics:
            val = all_results[name].get(m, 0)
            if m == 'PSNR':
                s = f'{val:.2f}'
            elif m == 'SSIM':
                s = f'{val:.4f}'
            else:
                s = f'{val:.3f}'

            if name == best[m]:
                s = r'\textbf{' + s + '}'
            elif name == second.get(m):
                s = r'\underline{' + s + '}'
            row.append(s)
        print(' & '.join(row) + r' \\')

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# =========================================================================
#  Main
# =========================================================================

def main():
    args = parse_args()

    # ---- Resolve method list ----
    if args.methods and args.pred_dirs:
        assert len(args.methods) == len(args.pred_dirs), \
            "--methods and --pred_dirs must have the same length"
        method_list = list(zip(args.methods, args.pred_dirs))
    elif args.method and args.pred_dir:
        method_list = [(args.method, args.pred_dir)]
    else:
        raise ValueError("Provide either (--method + --pred_dir) "
                         "or (--methods + --pred_dirs)")

    # ---- Build GT map ----
    test_stems = None
    if args.gt_index_dir:
        test_stems = get_test_stems(args.gt_index_dir, args.test_split_start)
        print(f"[GT] Using {len(test_stems)} test images "
              f"(split from index {args.test_split_start})")
    gt_map = collect_gt_files(args.gt_dir, test_stems)
    print(f"[GT] {len(gt_map)} ground-truth images from {args.gt_dir}\n")

    # ---- Evaluate each method ----
    all_results = OrderedDict()
    all_per_image = {}

    print(f"{'Method':20s} | {'PSNR':>6s} | {'SSIM':>6s} | "
          f"{'UCIQE':>6s} | {'UIQM':>6s} | n")
    print('-' * 72)

    for method_name, pred_dir in method_list:
        avg, per_img = evaluate_method(
            method_name, pred_dir, gt_map, args.img_size)
        if avg:
            all_results[method_name] = avg
            all_per_image[method_name] = per_img

    # ---- Output CSV ----
    if args.output_csv and all_results:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'PSNR', 'SSIM', 'UCIQE', 'UIQM'])
            for name, m in all_results.items():
                writer.writerow([name, f"{m['PSNR']:.4f}",
                                 f"{m['SSIM']:.6f}",
                                 f"{m['UCIQE']:.4f}",
                                 f"{m['UIQM']:.4f}"])
        print(f"\nCSV saved to: {args.output_csv}")

    # ---- Output JSON ----
    if args.output_json and all_per_image:
        with open(args.output_json, 'w') as f:
            json.dump({
                'summary': {k: v for k, v in all_results.items()},
                'per_image': all_per_image,
            }, f, indent=2)
        print(f"JSON saved to: {args.output_json}")

    # ---- LaTeX table ----
    if args.latex and all_results:
        print_latex_table(all_results)


if __name__ == '__main__':
    main()
