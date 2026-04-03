"""
plot_sensitivity.py — Visualize hyperparameter sensitivity results.

Reads the CSV produced by run_sensitivity.sh and generates:
  1. A PSNR heatmap  (λ₂ vs λ₃)
  2. An SSIM heatmap  (λ₂ vs λ₃)
  3. A 3D bar chart of PSNR across the grid

All figures are saved at 300 DPI, suitable for academic papers.

Usage:
    python plot_sensitivity.py --results_csv sensitivity_results.csv
    python plot_sensitivity.py --results_csv sensitivity_results.csv --output_dir ./figures
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)
import matplotlib.ticker as ticker


# =========================================================================
#  Shared style — academic / publication-ready
# =========================================================================

FONT_CFG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
}
plt.rcParams.update(FONT_CFG)


# =========================================================================
#  Heatmap generator
# =========================================================================

def plot_heatmap(pivot_df, metric_name, save_path):
    """
    Draw a 2D heatmap with annotated cell values.

    Args:
        pivot_df:    pandas pivot table (rows=lambda_phy, cols=lambda_freq)
        metric_name: 'PSNR (dB)' or 'SSIM'
        save_path:   output file path
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    data = pivot_df.values
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto',
                   origin='lower', interpolation='nearest')

    # Axis labels
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot_df.columns])
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([f'{v:.1f}' for v in pivot_df.index])

    ax.set_xlabel(r'$\lambda_3$ (Frequency Loss Weight)')
    ax.set_ylabel(r'$\lambda_2$ (Physics Loss Weight)')
    ax.set_title(f'{metric_name} Sensitivity')

    # Annotate each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Choose text colour for contrast
            thresh = data.min() + 0.6 * (data.max() - data.min())
            color = 'white' if val > thresh else 'black'
            fmt = f'{val:.2f}' if 'PSNR' in metric_name else f'{val:.4f}'
            ax.text(j, i, fmt, ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(metric_name)

    # Mark the best cell with a rectangle
    best_idx = np.unravel_index(data.argmax(), data.shape)
    rect = plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5),
                          1, 1, linewidth=2.5,
                          edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved heatmap: {save_path}")


# =========================================================================
#  3D bar chart
# =========================================================================

def plot_3d_bars(pivot_df, metric_name, save_path):
    """
    Draw a 3D bar chart of metric values over the (λ₂, λ₃) grid.
    """
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    data = pivot_df.values
    phy_vals = pivot_df.index.values
    freq_vals = pivot_df.columns.values

    xpos, ypos, zpos, dx, dy, dz, colors = [], [], [], [], [], [], []
    vmin, vmax = data.min(), data.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')

    bar_width_x = 0.6
    bar_width_y = 0.6

    for i, lp in enumerate(phy_vals):
        for j, lf in enumerate(freq_vals):
            xpos.append(j)
            ypos.append(i)
            zpos.append(0)
            dx.append(bar_width_x)
            dy.append(bar_width_y)
            dz.append(data[i, j])
            colors.append(cmap(norm(data[i, j])))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,
             edgecolor='grey', linewidth=0.4, alpha=0.92)

    ax.set_xticks(np.arange(len(freq_vals)) + bar_width_x / 2)
    ax.set_xticklabels([f'{v:.2f}' for v in freq_vals])
    ax.set_yticks(np.arange(len(phy_vals)) + bar_width_y / 2)
    ax.set_yticklabels([f'{v:.1f}' for v in phy_vals])

    ax.set_xlabel(r'$\lambda_3$ (Freq.)')
    ax.set_ylabel(r'$\lambda_2$ (Phys.)')
    ax.set_zlabel(metric_name)
    ax.set_title(f'{metric_name} vs. Loss Weights', pad=12)

    # Raise z-axis floor slightly for aesthetics
    z_margin = (vmax - vmin) * 0.05
    ax.set_zlim(max(0, vmin - z_margin), vmax + z_margin)

    ax.view_init(elev=28, azim=-50)

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(metric_name)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved 3D bar chart: {save_path}")


# =========================================================================
#  Combined summary figure (side-by-side PSNR + SSIM heatmaps)
# =========================================================================

def plot_combined(psnr_pivot, ssim_pivot, save_path):
    """
    Two-panel heatmap figure for direct paper inclusion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    for ax, pivot, label in [
        (axes[0], psnr_pivot, 'PSNR (dB)'),
        (axes[1], ssim_pivot, 'SSIM'),
    ]:
        data = pivot.values
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto',
                       origin='lower', interpolation='nearest')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.1f}' for v in pivot.index])
        ax.set_xlabel(r'$\lambda_3$ (Frequency)')
        ax.set_ylabel(r'$\lambda_2$ (Physics)')
        ax.set_title(label)

        # Annotate
        thresh = data.min() + 0.6 * (data.max() - data.min())
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                fmt = f'{val:.2f}' if 'PSNR' in label else f'{val:.4f}'
                color = 'white' if val > thresh else 'black'
                ax.text(j, i, fmt, ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

        # Best cell
        best = np.unravel_index(data.argmax(), data.shape)
        rect = plt.Rectangle((best[1] - 0.5, best[0] - 0.5),
                              1, 1, lw=2.5, ec='lime', fc='none')
        ax.add_patch(rect)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved combined figure: {save_path}")


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot hyperparameter sensitivity results')
    parser.add_argument('--results_csv', type=str,
                        default='sensitivity_results.csv',
                        help='CSV from run_sensitivity.sh')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Directory to save figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load results ----
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} grid-search results from {args.results_csv}")
    print(df.to_string(index=False))
    print()

    # Pivot tables
    psnr_pivot = df.pivot_table(
        values='best_psnr', index='lambda_phy', columns='lambda_freq')
    ssim_pivot = df.pivot_table(
        values='best_ssim', index='lambda_phy', columns='lambda_freq')

    # ---- Generate figures ----
    plot_heatmap(psnr_pivot, 'PSNR (dB)',
                 os.path.join(args.output_dir, 'sensitivity_psnr_heatmap.pdf'))
    plot_heatmap(ssim_pivot, 'SSIM',
                 os.path.join(args.output_dir, 'sensitivity_ssim_heatmap.pdf'))
    plot_3d_bars(psnr_pivot, 'PSNR (dB)',
                 os.path.join(args.output_dir, 'sensitivity_psnr_3d.pdf'))
    plot_combined(psnr_pivot, ssim_pivot,
                  os.path.join(args.output_dir, 'sensitivity_combined.pdf'))

    # ---- Print best config ----
    best_row = df.loc[df['best_psnr'].idxmax()]
    print(f"\nOptimal configuration:")
    print(f"  lambda_phy  = {best_row['lambda_phy']}")
    print(f"  lambda_freq = {best_row['lambda_freq']}")
    print(f"  PSNR        = {best_row['best_psnr']:.2f} dB")
    print(f"  SSIM        = {best_row['best_ssim']:.4f}")


if __name__ == '__main__':
    main()
