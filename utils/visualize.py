"""
Visualization utilities for paper figures.

Generates:
  - Side-by-side visual quality comparisons (Fig. 2 in paper)
  - Transmission map heatmaps
  - Ablation comparison grids
  - Metric bar charts
"""

import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


def tensor_to_numpy(t):
    """(C,H,W) tensor [0,1] -> (H,W,C) numpy uint8."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().clamp(0, 1).numpy()
    if t.shape[0] == 3:
        t = t.transpose(1, 2, 0)
    return (t * 255).astype(np.uint8)


def make_colormap(gray_np, colormap='jet'):
    """Convert (H,W) float32 [0,1] to colored heatmap (H,W,3) uint8."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    colored = cmap(gray_np)[..., :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def transmission_heatmap(t_c):
    """
    Visualize 3-channel transmission map as RGB heatmap.
    
    Args:
        t_c: (3, H, W) tensor or numpy
    Returns:
        (H, W, 3) numpy uint8 — colored heatmap of mean transmission
    """
    if isinstance(t_c, torch.Tensor):
        t_c = t_c.detach().cpu().numpy()
    t_mean = t_c.mean(axis=0)  # (H, W)
    t_norm = (t_mean - t_mean.min()) / (t_mean.max() - t_mean.min() + 1e-8)
    return make_colormap(t_norm, 'viridis')


def channel_transmission_vis(t_c):
    """
    Visualize per-channel (R/G/B) transmission maps side-by-side.
    Useful to show wideband attenuation differences.
    
    Args:
        t_c: (3, H, W) tensor
    Returns:
        (H, W*3, 3) numpy uint8
    """
    if isinstance(t_c, torch.Tensor):
        t_c = t_c.detach().cpu().numpy()
    
    channel_maps = []
    colormaps = ['Reds', 'Greens', 'Blues']
    for c in range(3):
        ch = t_c[c]
        ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        channel_maps.append(make_colormap(ch_norm, colormaps[c]))
    
    return np.concatenate(channel_maps, axis=1)


def add_label(img_np, text, position='bottom', fontsize=16, color=(255, 255, 255)):
    """Add text label to image."""
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == 'bottom':
        xy = ((img.width - tw) // 2, img.height - th - 8)
    elif position == 'top':
        xy = ((img.width - tw) // 2, 8)
    else:
        xy = position

    # Background rectangle for readability
    pad = 4
    draw.rectangle(
        [xy[0] - pad, xy[1] - pad, xy[0] + tw + pad, xy[1] + th + pad],
        fill=(0, 0, 0, 180)
    )
    draw.text(xy, text, fill=color, font=font)
    return np.array(img)


def make_comparison_figure(images_dict, save_path=None, title=None):
    """
    Create side-by-side comparison figure for the paper.
    
    Args:
        images_dict: OrderedDict of {label: (C,H,W) tensor or (H,W,C) numpy}
                     e.g., {'Input': img, 'DCP': img, 'Ours': img, 'GT': img}
        save_path:   where to save
        title:       optional super-title
    Returns:
        concatenated numpy image
    """
    panels = []
    for label, img in images_dict.items():
        img_np = tensor_to_numpy(img) if isinstance(img, torch.Tensor) else img
        # Ensure all same size
        if panels:
            h, w = panels[0].shape[:2]
            if img_np.shape[:2] != (h, w):
                img_pil = Image.fromarray(img_np).resize((w, h), Image.LANCZOS)
                img_np = np.array(img_pil)
        img_np = add_label(img_np, label, position='bottom', fontsize=14)
        panels.append(img_np)

    # Horizontal concatenation with 2px white separator
    separator = np.ones((panels[0].shape[0], 2, 3), dtype=np.uint8) * 255
    result = panels[0]
    for p in panels[1:]:
        result = np.concatenate([result, separator, p], axis=1)

    if title:
        # Add title bar at top
        h, w = result.shape[:2]
        title_bar = np.ones((30, w, 3), dtype=np.uint8) * 255
        title_img = Image.fromarray(title_bar)
        draw = ImageDraw.Draw(title_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, 5), title, fill=(0, 0, 0), font=font)
        result = np.concatenate([np.array(title_img), result], axis=0)

    if save_path:
        Image.fromarray(result).save(save_path)
        print(f"Saved comparison figure: {save_path}")
    return result


def make_ablation_grid(ablation_dict, save_path=None):
    """
    Create ablation comparison grid.
    
    Args:
        ablation_dict: {
            'image_name': {
                'Input': tensor, 'w/o Physics': tensor, 
                'w/o FFT': tensor, 'w/o SNR Gate': tensor,
                'Full (Ours)': tensor, 'GT': tensor
            }
        }
    """
    rows = []
    for img_name, methods in ablation_dict.items():
        row = make_comparison_figure(methods, title=img_name)
        rows.append(row)

    # Stack rows vertically
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    
    sep = np.ones((3, max_w, 3), dtype=np.uint8) * 200
    result = padded[0]
    for r in padded[1:]:
        result = np.concatenate([result, sep, r], axis=0)

    if save_path:
        Image.fromarray(result).save(save_path)
        print(f"Saved ablation grid: {save_path}")
    return result


def make_metric_table_latex(results_dict, metrics=('PSNR', 'SSIM', 'UCIQE', 'UIQM')):
    """
    Generate LaTeX table for quantitative comparison.
    
    Args:
        results_dict: {
            'DCP':       {'PSNR': 16.2, 'SSIM': 0.72, ...},
            'WaterNet':  {'PSNR': 19.5, 'SSIM': 0.82, ...},
            'Ours':      {'PSNR': 23.1, 'SSIM': 0.91, ...},
        }
    Returns:
        LaTeX string
    """
    methods = list(results_dict.keys())

    # Find best values per metric
    best = {}
    second_best = {}
    for m in metrics:
        vals = [(name, results_dict[name].get(m, 0)) for name in methods]
        vals.sort(key=lambda x: x[1], reverse=True)
        best[m] = vals[0][0]
        if len(vals) > 1:
            second_best[m] = vals[1][0]

    # Build table
    n_metrics = len(metrics)
    col_spec = 'l' + 'c' * n_metrics
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Quantitative comparison on the UIEB test set. '
        r'\textbf{Bold}: best, \underline{underline}: second best.}',
        r'\label{tab:comparison}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        'Method & ' + ' & '.join(metrics) + r' \\',
        r'\midrule',
    ]

    for name in methods:
        row = [name]
        for m in metrics:
            val = results_dict[name].get(m, 0)
            if m in ('PSNR',):
                s = f'{val:.2f}'
            elif m in ('SSIM',):
                s = f'{val:.4f}'
            else:
                s = f'{val:.3f}'
            
            if name == best[m]:
                s = r'\textbf{' + s + '}'
            elif name == second_best.get(m):
                s = r'\underline{' + s + '}'
            row.append(s)
        lines.append(' & '.join(row) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def make_ablation_table_latex(ablation_results):
    """
    Generate LaTeX table for ablation study.
    
    Args:
        ablation_results: {
            'Full (Ours)':  {'PSNR': 23.1, 'SSIM': 0.91},
            'w/o Physics':  {'PSNR': 20.5, 'SSIM': 0.85},
            'w/o FFT':      {'PSNR': 21.2, 'SSIM': 0.87},
            'w/o SNR Gate': {'PSNR': 22.0, 'SSIM': 0.89},
        }
    """
    metrics = ('PSNR', 'SSIM', 'UCIQE', 'UIQM')
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Ablation study results on the UIEB test set.}',
        r'\label{tab:ablation}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        'Configuration & PSNR$\uparrow$ & SSIM$\uparrow$ & UCIQE$\uparrow$ & UIQM$\uparrow$ \\\\',
        r'\midrule',
    ]

    for name, vals in ablation_results.items():
        row = [name]
        for m in metrics:
            v = vals.get(m, 0)
            fmt = f'{v:.2f}' if m == 'PSNR' else f'{v:.4f}'
            if name == 'Full (Ours)':
                fmt = r'\textbf{' + fmt + '}'
            row.append(fmt)
        lines.append(' & '.join(row) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def plot_loss_curves(log_path, save_path):
    """
    Plot training loss curves from a JSON log file.
    
    Args:
        log_path: path to training log (one JSON per line)
        save_path: output image path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json

    epochs, losses = [], {'total': [], 'rec': [], 'phy': [], 'freq': [], 'edge': []}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            epochs.append(entry['epoch'])
            for k in losses:
                losses[k].append(entry.get(k, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    axes[0].plot(epochs, losses['total'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Component losses
    for k in ['rec', 'phy', 'freq', 'edge']:
        axes[1].plot(epochs, losses[k], label=f'L_{k}', linewidth=1.2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves: {save_path}")
