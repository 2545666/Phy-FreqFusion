"""
Benchmark script: run all methods, compute metrics, generate paper tables/figures.

Usage:
    python benchmark.py \
        --data_root ./data/UIEB \
        --ours_ckpt ./checkpoints/full_best/best.pth \
        --ablation_dir ./checkpoints \
        --output_dir ./paper_results
"""

import os
import sys
import argparse
import json
import time
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.phy_freqfusion import (
    PhyFreqFusion, PhyFreqFusion_NoPhysics,
    PhyFreqFusion_NoFFT, PhyFreqFusion_NoSNRGate
)
from datasets.uie_dataset import UIEBDataset, EUVPDataset
from utils.metrics import MetricTracker, calc_psnr, calc_ssim, calc_uciqe, calc_uiqm
from utils.visualize import (
    make_comparison_figure, make_ablation_grid,
    make_metric_table_latex, make_ablation_table_latex,
    transmission_heatmap, channel_transmission_vis,
    tensor_to_numpy
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--dataset', type=str, default='uieb', choices=['uieb', 'euvp'])
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--base_ch', type=int, default=64)
    # Checkpoints
    p.add_argument('--ours_ckpt', type=str, default=None,
                   help='Path to full model checkpoint')
    p.add_argument('--ablation_dir', type=str, default=None,
                   help='Dir containing ablation variant checkpoints')
    # Baseline results (pre-computed JSONs from other methods)
    p.add_argument('--baseline_results', type=str, default=None,
                   help='JSON file with baseline metrics: {"DCP": {"PSNR":...}, ...}')
    p.add_argument('--baseline_images', type=str, default=None,
                   help='Dir with baseline outputs: baselines/{method}/{image}.png')
    # Output
    p.add_argument('--output_dir', type=str, default='./paper_results')
    p.add_argument('--select_images', type=str, nargs='*', default=None,
                   help='Specific image filenames for visual comparison')
    return p.parse_args()


def load_model(variant, ckpt_path, base_ch=64, device='cpu'):
    """Load a model variant from checkpoint."""
    model_map = {
        'full': PhyFreqFusion,
        'no_physics': PhyFreqFusion_NoPhysics,
        'no_fft': PhyFreqFusion_NoFFT,
        'no_snr_gate': PhyFreqFusion_NoSNRGate,
    }
    model = model_map[variant](base_ch=base_ch, lap_levels=3)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


def evaluate_model(model, dataloader, device):
    """Run evaluation, return per-image and aggregate metrics."""
    tracker = MetricTracker()
    per_image = {}
    times = []

    with torch.no_grad():
        for batch in dataloader:
            inp = batch['input'].to(device)
            gt = batch['target'].to(device)
            fname = batch['filename'][0]

            t0 = time.time()
            out = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)

            pred = out['J_pred'][0]
            tracker.update(pred, gt[0])

            per_image[fname] = {
                'PSNR': calc_psnr(pred, gt[0]),
                'SSIM': calc_ssim(pred.unsqueeze(0), gt[0].unsqueeze(0)),
                'UCIQE': calc_uciqe(pred),
                'UIQM': calc_uiqm(pred),
            }

    summary = tracker.summary()
    summary['Avg_Time_ms'] = np.mean(times) * 1000
    summary['Params_M'] = sum(p.numel() for p in model.parameters()) / 1e6
    return summary, per_image


def run_full_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'transmission'), exist_ok=True)

    # Build test dataset
    if args.dataset == 'uieb':
        test_ds = UIEBDataset(args.data_root, 'test', args.img_size, augment=False)
    else:
        test_ds = EUVPDataset(args.data_root, 'val', args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    all_results = OrderedDict()

    # -------------------------------------------------------
    # 1. Load baseline results (pre-computed by other methods)
    # -------------------------------------------------------
    if args.baseline_results and os.path.exists(args.baseline_results):
        with open(args.baseline_results, 'r') as f:
            baselines = json.load(f)
        all_results.update(baselines)
        print(f"[Baselines] Loaded {len(baselines)} methods from {args.baseline_results}")
    else:
        # Placeholder baseline results for table generation demo
        all_results['DCP [He 2011]'] = {'PSNR': 14.52, 'SSIM': 0.6438, 'UCIQE': 0.542, 'UIQM': 2.85}
        all_results['UDCP [Drews 2013]'] = {'PSNR': 13.87, 'SSIM': 0.5912, 'UCIQE': 0.498, 'UIQM': 2.61}
        all_results['WaterNet [Li 2020]'] = {'PSNR': 19.15, 'SSIM': 0.8023, 'UCIQE': 0.582, 'UIQM': 3.12}
        all_results['FUnIE-GAN [Islam 2020]'] = {'PSNR': 18.43, 'SSIM': 0.7815, 'UCIQE': 0.571, 'UIQM': 3.08}
        all_results['U-Shape Trans. [Peng 2023]'] = {'PSNR': 21.38, 'SSIM': 0.8574, 'UCIQE': 0.601, 'UIQM': 3.34}
        print("[Baselines] Using placeholder values — replace with actual numbers")

    # -------------------------------------------------------
    # 2. Evaluate our full model
    # -------------------------------------------------------
    if args.ours_ckpt and os.path.exists(args.ours_ckpt):
        print("\n[Ours] Evaluating full model...")
        model = load_model('full', args.ours_ckpt, args.base_ch, device)
        ours_summary, ours_per_image = evaluate_model(model, test_loader, device)
        all_results['Ours'] = ours_summary

        print(f"  PSNR={ours_summary['PSNR']:.2f}  SSIM={ours_summary['SSIM']:.4f}  "
              f"UCIQE={ours_summary['UCIQE']:.4f}  UIQM={ours_summary['UIQM']:.4f}")
        print(f"  Params={ours_summary['Params_M']:.2f}M  "
              f"Time={ours_summary['Avg_Time_ms']:.1f}ms")

        # Save transmission maps for selected images
        print("\n[Vis] Generating transmission visualizations...")
        with torch.no_grad():
            for batch in test_loader:
                fname = batch['filename'][0]
                if args.select_images and fname not in args.select_images:
                    continue
                inp = batch['input'].to(device)
                out = model(inp)
                t_c = out['t_c'][0]

                # Mean transmission heatmap
                hmap = transmission_heatmap(t_c)
                Image.fromarray(hmap).save(
                    os.path.join(args.output_dir, 'transmission', 
                                 f'{os.path.splitext(fname)[0]}_trans_heat.png'))

                # Per-channel (R/G/B) transmission
                ch_vis = channel_transmission_vis(t_c)
                Image.fromarray(ch_vis).save(
                    os.path.join(args.output_dir, 'transmission',
                                 f'{os.path.splitext(fname)[0]}_trans_rgb.png'))

        # Save per-image metrics
        with open(os.path.join(args.output_dir, 'per_image_metrics.json'), 'w') as f:
            json.dump(ours_per_image, f, indent=2)

    # -------------------------------------------------------
    # 3. Ablation study
    # -------------------------------------------------------
    ablation_results = OrderedDict()
    ablation_variants = {
        'w/o Physics': 'no_physics',
        'w/o FFT': 'no_fft',
        'w/o SNR Gate': 'no_snr_gate',
    }

    if args.ablation_dir and os.path.isdir(args.ablation_dir):
        # Look for checkpoint files: {variant}_*/best.pth
        for label, variant in ablation_variants.items():
            ckpt_candidates = [
                os.path.join(args.ablation_dir, f'{variant}_best.pth'),
                os.path.join(args.ablation_dir, f'{variant}/best.pth'),
            ]
            # Also search for timestamped dirs
            for d in os.listdir(args.ablation_dir):
                if d.startswith(variant):
                    ckpt_candidates.append(
                        os.path.join(args.ablation_dir, d, 'best.pth'))

            ckpt_path = None
            for c in ckpt_candidates:
                if os.path.exists(c):
                    ckpt_path = c
                    break

            if ckpt_path:
                print(f"\n[Ablation] Evaluating {label} ({variant})...")
                m = load_model(variant, ckpt_path, args.base_ch, device)
                summary, _ = evaluate_model(m, test_loader, device)
                ablation_results[label] = summary
                print(f"  PSNR={summary['PSNR']:.2f}  SSIM={summary['SSIM']:.4f}")
            else:
                print(f"[Ablation] Checkpoint not found for {label}, skipping")

    # Add full model to ablation table
    if 'Ours' in all_results:
        ablation_results['Full (Ours)'] = all_results['Ours']
    
    # Reorder: variants first, then full
    ordered_ablation = OrderedDict()
    for k in ['w/o Physics', 'w/o FFT', 'w/o SNR Gate', 'Full (Ours)']:
        if k in ablation_results:
            ordered_ablation[k] = ablation_results[k]

    # -------------------------------------------------------
    # 4. Generate paper tables (LaTeX)
    # -------------------------------------------------------
    print("\n[Tables] Generating LaTeX tables...")

    # Comparison table
    comp_latex = make_metric_table_latex(all_results)
    comp_path = os.path.join(args.output_dir, 'tables', 'comparison_table.tex')
    with open(comp_path, 'w') as f:
        f.write(comp_latex)
    print(f"  Saved: {comp_path}")

    # Ablation table
    if ordered_ablation:
        abl_latex = make_ablation_table_latex(ordered_ablation)
        abl_path = os.path.join(args.output_dir, 'tables', 'ablation_table.tex')
        with open(abl_path, 'w') as f:
            f.write(abl_latex)
        print(f"  Saved: {abl_path}")

    # -------------------------------------------------------
    # 5. Generate visual comparisons
    # -------------------------------------------------------
    if args.ours_ckpt and os.path.exists(args.ours_ckpt):
        print("\n[Figures] Generating visual comparison figures...")
        model = load_model('full', args.ours_ckpt, args.base_ch, device)
        
        with torch.no_grad():
            count = 0
            for batch in test_loader:
                fname = batch['filename'][0]
                if args.select_images and fname not in args.select_images:
                    continue
                if count >= 6:  # max 6 comparison figures
                    break

                inp = batch['input'].to(device)
                gt = batch['target'].to(device)
                out = model(inp)

                images = OrderedDict()
                images['Input'] = inp[0].cpu()
                
                # Load baseline images if available
                if args.baseline_images:
                    for method in ['DCP', 'WaterNet', 'U-Shape']:
                        bpath = os.path.join(args.baseline_images, method, fname)
                        if os.path.exists(bpath):
                            bimg = Image.open(bpath).convert('RGB').resize(
                                (args.img_size, args.img_size))
                            images[method] = np.array(bimg)

                images['Ours'] = out['J_pred'][0].cpu()
                images['GT'] = gt[0].cpu()

                save_path = os.path.join(args.output_dir, 'figures',
                                         f'comparison_{os.path.splitext(fname)[0]}.png')
                make_comparison_figure(images, save_path=save_path)
                count += 1

    # -------------------------------------------------------
    # 6. Summary report
    # -------------------------------------------------------
    report_path = os.path.join(args.output_dir, 'benchmark_summary.json')
    report = {
        'comparison': {k: v for k, v in all_results.items()},
        'ablation': {k: v for k, v in ordered_ablation.items()},
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Benchmark Complete")
    print(f"{'='*60}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  - tables/comparison_table.tex")
    print(f"  - tables/ablation_table.tex")
    print(f"  - figures/comparison_*.png")
    print(f"  - transmission/*_trans_*.png")
    print(f"  - benchmark_summary.json")
    print(f"{'='*60}")

    # Print comparison table to console
    print("\n--- Quantitative Comparison ---")
    print(f"{'Method':<30} {'PSNR':>8} {'SSIM':>8} {'UCIQE':>8} {'UIQM':>8}")
    print('-' * 66)
    for name, vals in all_results.items():
        print(f"{name:<30} {vals.get('PSNR',0):>8.2f} {vals.get('SSIM',0):>8.4f} "
              f"{vals.get('UCIQE',0):>8.4f} {vals.get('UIQM',0):>8.4f}")

    if ordered_ablation:
        print("\n--- Ablation Study ---")
        print(f"{'Configuration':<25} {'PSNR':>8} {'SSIM':>8} {'UCIQE':>8} {'UIQM':>8}")
        print('-' * 60)
        for name, vals in ordered_ablation.items():
            print(f"{name:<25} {vals.get('PSNR',0):>8.2f} {vals.get('SSIM',0):>8.4f} "
                  f"{vals.get('UCIQE',0):>8.4f} {vals.get('UIQM',0):>8.4f}")


if __name__ == '__main__':
    args = parse_args()
    run_full_benchmark(args)
