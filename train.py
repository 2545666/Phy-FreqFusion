"""
Training script for Phy-FreqFusion.

Usage:
    python train.py --data_root ./data/UIEB --epochs 200 --batch_size 8 --lr 2e-4

Supports: UIEB / EUVP datasets, mixed precision, cosine LR scheduling.
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.phy_freqfusion import (
    PhyFreqFusion, PhyFreqFusion_NoPhysics,
    PhyFreqFusion_NoFFT, PhyFreqFusion_NoSNRGate
)
from losses.losses import PhyFreqFusionLoss
from datasets.uie_dataset import UIEBDataset, EUVPDataset
from utils.metrics import calc_psnr, calc_ssim


def parse_args():
    p = argparse.ArgumentParser(description='Train Phy-FreqFusion')
    # Data
    p.add_argument('--data_root', type=str, required=True,
                   help='Root dir of UIEB or EUVP dataset')
    p.add_argument('--dataset', type=str, default='uieb',
                   choices=['uieb', 'euvp'])
    p.add_argument('--img_size', type=int, default=256)
    # Model
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--lap_levels', type=int, default=3)
    p.add_argument('--variant', type=str, default='full',
                   choices=['full', 'no_physics', 'no_fft', 'no_snr_gate'],
                   help='Model variant for ablation study')
    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lambda_phy', type=float, default=0.5)
    p.add_argument('--lambda_freq', type=float, default=0.1)
    p.add_argument('--lambda_edge', type=float, default=0.2)
    p.add_argument('--amp', action='store_true', help='Mixed precision')
    p.add_argument('--num_workers', type=int, default=4)
    # Logging
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def build_model(args):
    """Build model variant."""
    if args.variant == 'full':
        return PhyFreqFusion(base_ch=args.base_ch, lap_levels=args.lap_levels)
    elif args.variant == 'no_physics':
        return PhyFreqFusion_NoPhysics(base_ch=args.base_ch, lap_levels=args.lap_levels)
    elif args.variant == 'no_fft':
        return PhyFreqFusion_NoFFT(base_ch=args.base_ch, lap_levels=args.lap_levels)
    elif args.variant == 'no_snr_gate':
        return PhyFreqFusion_NoSNRGate(base_ch=args.base_ch, lap_levels=args.lap_levels)


def build_dataloader(args):
    """Build train/val dataloaders."""
    if args.dataset == 'uieb':
        train_ds = UIEBDataset(args.data_root, 'train', args.img_size, augment=True)
        val_ds = UIEBDataset(args.data_root, 'test', args.img_size, augment=False)
    else:
        train_ds = EUVPDataset(args.data_root, 'train', args.img_size, augment=True)
        val_ds = EUVPDataset(args.data_root, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    return train_loader, val_loader


def validate(model, val_loader, device):
    """Run validation; return average PSNR and SSIM."""
    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch['input'].to(device)
            gt = batch['target'].to(device)
            out = model(inp)
            pred = out['J_pred']
            for i in range(pred.size(0)):
                psnr_sum += calc_psnr(pred[i], gt[i])
                ssim_sum += calc_ssim(pred[i].unsqueeze(0), gt[i].unsqueeze(0))
                count += 1
    model.train()
    return psnr_sum / max(count, 1), ssim_sum / max(count, 1)


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Config] Device: {device}, Variant: {args.variant}")
    print(f"[Config] Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}")

    # Save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{args.variant}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_loader, val_loader = build_dataloader(args)

    # Model
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Parameters: {n_params:.2f} M")

    # Loss
    criterion = PhyFreqFusionLoss(
        lambda_phy=args.lambda_phy,
        lambda_freq=args.lambda_freq,
        lambda_edge=args.lambda_edge,
    ).to(device)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler() if args.amp else None

    # Resume
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"[Resume] From epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    # ---- Training Loop ----
    print(f"\n{'='*60}")
    print(f"  Training Phy-FreqFusion ({args.variant})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        loss_accum = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            inp = batch['input'].to(device)
            gt = batch['target'].to(device)

            optimizer.zero_grad()

            if args.amp:
                with autocast():
                    output = model(inp)
                    loss, ld = criterion(output, inp, gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(inp)
                loss, ld = criterion(output, inp, gt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            epoch_loss += ld['total']
            for k, v in ld.items():
                loss_accum[k] = loss_accum.get(k, 0) + v

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in loss_accum.items()}
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {avg['total']:.4f} "
                      f"(rec={avg['rec']:.4f} phy={avg['phy']:.4f} "
                      f"freq={avg['freq']:.4f} edge={avg['edge']:.4f})")

        scheduler.step()

        # Epoch summary
        n_steps = len(train_loader)
        avg_loss = epoch_loss / n_steps
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"AvgLoss={avg_loss:.4f} LR={lr_now:.2e} Time={elapsed:.1f}s")

        # Validation
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            val_psnr, val_ssim = validate(model, val_loader, device)
            print(f"  >> Val PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

            # Save checkpoint
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': max(best_psnr, val_psnr),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }
            torch.save(ckpt, os.path.join(save_dir, 'latest.pth'))

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(ckpt, os.path.join(save_dir, 'best.pth'))
                print(f"  >> New best! PSNR={best_psnr:.2f}")

    print(f"\nTraining complete. Best PSNR={best_psnr:.2f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    train()
