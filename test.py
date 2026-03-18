"""
Evaluation script for Phy-FreqFusion.

Usage:
    # Full evaluation with metrics (requires GT)
    python test.py --data_root ./data/UIEB --checkpoint ./checkpoints/best.pth --save_images

    # Inference only (no GT needed)
    python test.py --input_dir ./test_images --checkpoint ./checkpoints/best.pth --mode infer
"""

import os
import sys
import argparse
import time

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.phy_freqfusion import (
    PhyFreqFusion, PhyFreqFusion_NoPhysics,
    PhyFreqFusion_NoFFT, PhyFreqFusion_NoSNRGate
)
from datasets.uie_dataset import UIEBDataset, EUVPDataset, TestDataset
from utils.metrics import MetricTracker


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Phy-FreqFusion')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--mode', type=str, default='eval', choices=['eval', 'infer'])
    # eval mode
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--dataset', type=str, default='uieb', choices=['uieb', 'euvp'])
    # infer mode
    p.add_argument('--input_dir', type=str, default=None)
    # shared
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--lap_levels', type=int, default=3)
    p.add_argument('--variant', type=str, default='full',
                   choices=['full', 'no_physics', 'no_fft', 'no_snr_gate'])
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--output_dir', type=str, default='./results')
    return p.parse_args()


def build_model(args, device):
    variants = {
        'full': PhyFreqFusion,
        'no_physics': PhyFreqFusion_NoPhysics,
        'no_fft': PhyFreqFusion_NoFFT,
        'no_snr_gate': PhyFreqFusion_NoSNRGate,
    }
    model_cls = variants[args.variant]
    if args.variant == 'no_physics':
        model = model_cls(base_ch=args.base_ch, lap_levels=args.lap_levels)
    else:
        model = model_cls(base_ch=args.base_ch, lap_levels=args.lap_levels)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"[Model] Loaded {args.variant} from {args.checkpoint}")
    if 'val_psnr' in ckpt:
        print(f"[Model] Checkpoint PSNR={ckpt['val_psnr']:.2f}, SSIM={ckpt.get('val_ssim', 0):.4f}")
    return model


def tensor_to_pil(t):
    """(C,H,W) tensor [0,1] -> PIL Image."""
    t = t.clamp(0, 1).detach().cpu()
    return TF.to_pil_image(t)


def save_visual(inp, pred, gt, t_c, filename, output_dir):
    """Save side-by-side comparison + transmission map."""
    os.makedirs(output_dir, exist_ok=True)
    
    name = os.path.splitext(filename)[0]
    tensor_to_pil(pred).save(os.path.join(output_dir, f'{name}_pred.png'))
    tensor_to_pil(inp).save(os.path.join(output_dir, f'{name}_input.png'))
    
    if gt is not None:
        tensor_to_pil(gt).save(os.path.join(output_dir, f'{name}_gt.png'))

    # Save transmission map (average across channels, normalized)
    t_vis = t_c.mean(dim=0).detach().cpu()  # (H, W)
    t_vis = (t_vis - t_vis.min()) / (t_vis.max() - t_vis.min() + 1e-8)
    t_pil = TF.to_pil_image(t_vis.unsqueeze(0))
    t_pil.save(os.path.join(output_dir, f'{name}_trans.png'))


def evaluate(args):
    """Full evaluation with metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)

    # Build dataset
    if args.dataset == 'uieb':
        ds = UIEBDataset(args.data_root, 'test', args.img_size, augment=False)
    else:
        ds = EUVPDataset(args.data_root, 'val', args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    tracker = MetricTracker()
    times = []

    print(f"\nEvaluating on {len(ds)} images...")
    with torch.no_grad():
        for batch in loader:
            inp = batch['input'].to(device)
            gt = batch['target'].to(device)
            fname = batch['filename'][0]

            t0 = time.time()
            out = model(inp)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.time() - t0)

            pred = out['J_pred'][0]
            tracker.update(pred, gt[0])

            if args.save_images:
                save_visual(inp[0], pred, gt[0], out['t_c'][0],
                           fname, args.output_dir)

    # Print results
    metrics = tracker.summary()
    avg_time = np.mean(times) * 1000  # ms

    print(f"\n{'='*50}")
    print(f"  Results: {args.variant} on {args.dataset.upper()}")
    print(f"{'='*50}")
    print(f"  PSNR:  {metrics['PSNR']:.2f} dB")
    print(f"  SSIM:  {metrics['SSIM']:.4f}")
    print(f"  UCIQE: {metrics['UCIQE']:.4f}")
    print(f"  UIQM:  {metrics['UIQM']:.4f}")
    print(f"  Avg Inference Time: {avg_time:.1f} ms")
    print(f"{'='*50}")

    return metrics


def infer(args):
    """Inference-only mode (no ground truth) + No-Reference metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)

    # TestDataset 专门读取没有任何分类、全是图片的文件夹
    ds = TestDataset(args.input_dir, args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nProcessing {len(ds)} images for Zero-Shot Generalization...")

    tracker = MetricTracker()
    times = []

    with torch.no_grad():
        for batch in loader:
            inp = batch['input'].to(device)
            fname = batch['filename'][0]

            t0 = time.time()
            out = model(inp)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.time() - t0)

            pred = out['J_pred'][0]
            
            # 关键修改：gt 传 None，触发仅计算 UCIQE 和 UIQM
            tracker.update(pred, gt=None)
            
            # 保存原图、预测图和透射率图的组合可视化
            save_visual(inp[0], pred, None, out['t_c'][0],
                       fname, args.output_dir)
            print(f"  Processed: {fname}")

    # 获取跑分并打印
    metrics = tracker.summary()
    avg_time = np.mean(times) * 1000

    print(f"\n{'='*55}")
    print(f"  Zero-Shot Generalization Results (No Ground Truth)")
    print(f"{'='*55}")
    print(f"  UCIQE: {metrics['UCIQE']:.4f} (Higher is better)")
    print(f"  UIQM:  {metrics['UIQM']:.4f} (Higher is better)")
    print(f"  Avg Inference Time: {avg_time:.1f} ms")
    print(f"{'='*55}")
    print(f"Enhanced images saved to: {args.output_dir}")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'eval':
        assert args.data_root, "--data_root required for eval mode"
        evaluate(args)
    else:
        assert args.input_dir, "--input_dir required for infer mode"
        infer(args)
