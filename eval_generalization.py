"""
eval_generalization.py — Cross-dataset generalization evaluation.

Runs a pre-trained Phy-FreqFusion checkpoint on third-party datasets
(LSUI, RUIE, or any folder of images) and reports metrics.

  • LSUI (paired)  → full-reference (PSNR, SSIM) + no-reference (UCIQE, UIQM)
  • RUIE (unpaired) → no-reference only (UCIQE, UIQM)

Usage examples:
  # Full-reference evaluation on LSUI
  python eval_generalization.py \
      --checkpoint checkpoints/best.pth \
      --dataset lsui \
      --data_root ./data/LSUI \
      --save_images --output_dir ./results/lsui

  # No-reference evaluation on RUIE (all subsets)
  python eval_generalization.py \
      --checkpoint checkpoints/best.pth \
      --dataset ruie \
      --data_root ./data/RUIE

  # No-reference evaluation on a specific RUIE subset
  python eval_generalization.py \
      --checkpoint checkpoints/best.pth \
      --dataset ruie \
      --data_root ./data/RUIE \
      --ruie_subset UCCS

  # Also supports EUVP for completeness
  python eval_generalization.py \
      --checkpoint checkpoints/best.pth \
      --dataset euvp \
      --data_root ./data/EUVP/underwater_dark
"""

import os
import sys
import argparse
import time
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.phy_freqfusion import (
    PhyFreqFusion, PhyFreqFusion_NoPhysics,
    PhyFreqFusion_NoFFT, PhyFreqFusion_NoSNRGate
)
from datasets.uie_dataset import (
    LSUIDataset, RUIEDataset, EUVPDataset, UIEBDataset
)
from utils.metrics import MetricTracker


# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate Phy-FreqFusion generalization on external datasets')

    # Required
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained .pth checkpoint')
    p.add_argument('--data_root', type=str, required=True,
                   help='Root directory of the target dataset')
    p.add_argument('--dataset', type=str, required=True,
                   choices=['lsui', 'ruie', 'euvp', 'uieb'],
                   help='Which dataset to evaluate on')

    # RUIE-specific
    p.add_argument('--ruie_subset', type=str, default=None,
                   choices=['UCCS', 'UIQS', 'UHTS'],
                   help='Evaluate only a specific RUIE subset (default: all)')

    # Model config
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--lap_levels', type=int, default=3)
    p.add_argument('--variant', type=str, default='full',
                   choices=['full', 'no_physics', 'no_fft', 'no_snr_gate'])

    # Output
    p.add_argument('--save_images', action='store_true',
                   help='Save enhanced images to --output_dir')
    p.add_argument('--output_dir', type=str, default='./results/generalization')
    p.add_argument('--max_samples', type=int, default=None,
                   help='Cap number of evaluation images (for quick tests)')
    p.add_argument('--save_json', action='store_true',
                   help='Write per-image metrics to a JSON file')

    return p.parse_args()


# =========================================================================
#  Helpers
# =========================================================================

def build_model(args, device):
    """Instantiate model and load checkpoint weights."""
    variants = {
        'full': PhyFreqFusion,
        'no_physics': PhyFreqFusion_NoPhysics,
        'no_fft': PhyFreqFusion_NoFFT,
        'no_snr_gate': PhyFreqFusion_NoSNRGate,
    }
    model = variants[args.variant](
        base_ch=args.base_ch, lap_levels=args.lap_levels)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    print(f"[Model] Loaded {args.variant} from {args.checkpoint}")
    if 'val_psnr' in ckpt:
        print(f"[Model] Checkpoint val PSNR={ckpt['val_psnr']:.2f}, "
              f"SSIM={ckpt.get('val_ssim', 0):.4f}")
    return model


def build_dataset(args):
    """Build the appropriate dataset loader."""
    if args.dataset == 'lsui':
        return LSUIDataset(args.data_root, img_size=args.img_size,
                           max_samples=args.max_samples)
    elif args.dataset == 'ruie':
        return RUIEDataset(args.data_root, subset=args.ruie_subset,
                           img_size=args.img_size,
                           max_samples=args.max_samples)
    elif args.dataset == 'euvp':
        return EUVPDataset(args.data_root, split='val',
                           img_size=args.img_size, augment=False)
    elif args.dataset == 'uieb':
        return UIEBDataset(args.data_root, split='test',
                           img_size=args.img_size, augment=False)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def tensor_to_pil(t):
    """(C,H,W) tensor [0,1] -> PIL Image."""
    return TF.to_pil_image(t.clamp(0, 1).detach().cpu())


def has_ground_truth(dataset_name):
    """Determine if a dataset provides paired ground-truth images."""
    return dataset_name in ('lsui', 'euvp', 'uieb')


# =========================================================================
#  Main evaluation loop
# =========================================================================

def evaluate_generalization(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)
    ds = build_dataset(args)

    paired = has_ground_truth(args.dataset)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    tracker = MetricTracker()
    times = []
    per_image_metrics = []

    tag = args.dataset.upper()
    if args.dataset == 'ruie' and args.ruie_subset:
        tag += f'/{args.ruie_subset}'

    print(f"\n{'='*60}")
    print(f"  Generalization Evaluation: {args.variant} on {tag}")
    print(f"  Images: {len(ds)}  |  Paired GT: {paired}")
    print(f"{'='*60}\n")

    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            inp = batch['input'].to(device)
            gt = batch.get('target', None)
            if gt is not None:
                gt = gt.to(device)
            fname = batch['filename'][0]

            # Inference + timing
            t0 = time.time()
            out = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)

            pred = out['J_pred'][0]

            # Update tracker (handles None gt for no-reference)
            tracker.update(pred, gt[0] if gt is not None else None)

            # Optional: save enhanced image
            if args.save_images:
                tensor_to_pil(pred).save(
                    os.path.join(args.output_dir, fname))

            # Per-image record (for JSON export)
            if args.save_json:
                from utils.metrics import calc_psnr, calc_ssim, calc_uciqe, calc_uiqm
                entry = {'filename': fname}
                if gt is not None:
                    entry['PSNR'] = calc_psnr(pred, gt[0])
                    entry['SSIM'] = calc_ssim(
                        pred.unsqueeze(0), gt[0].unsqueeze(0))
                entry['UCIQE'] = calc_uciqe(pred)
                entry['UIQM'] = calc_uiqm(pred)
                per_image_metrics.append(entry)

    # ---- Aggregate results ----
    metrics = tracker.summary()
    avg_time = np.mean(times) * 1000  # ms

    print(f"\n{'='*60}")
    print(f"  Results: {args.variant} → {tag}")
    print(f"{'='*60}")
    if paired:
        print(f"  PSNR:  {metrics.get('PSNR', 0):.2f} dB")
        print(f"  SSIM:  {metrics.get('SSIM', 0):.4f}")
    print(f"  UCIQE: {metrics['UCIQE']:.4f}")
    print(f"  UIQM:  {metrics['UIQM']:.4f}")
    print(f"  Avg Inference Time: {avg_time:.1f} ms")
    print(f"  Total Images: {len(ds)}")
    print(f"{'='*60}")

    # ---- Save JSON log ----
    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(
            args.output_dir, f'metrics_{args.dataset}.json')
        summary = {
            'dataset': tag,
            'variant': args.variant,
            'checkpoint': args.checkpoint,
            'num_images': len(ds),
            'avg_metrics': {k: float(v) for k, v in metrics.items()},
            'avg_inference_ms': float(avg_time),
            'per_image': per_image_metrics,
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Per-image metrics saved to: {json_path}")

    if args.save_images:
        print(f"Enhanced images saved to: {args.output_dir}")

    return metrics


# =========================================================================

if __name__ == '__main__':
    args = parse_args()
    evaluate_generalization(args)
