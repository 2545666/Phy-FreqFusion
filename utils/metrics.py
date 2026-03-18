"""
Evaluation metrics for UIE:
  - Full-reference:  PSNR, SSIM
  - No-reference:    UCIQE, UIQM
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ===================== Full-Reference Metrics =====================

def calc_psnr(pred, gt, max_val=1.0):
    """
    PSNR between two tensors or numpy arrays.
    Higher is better.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(max_val ** 2 / mse)


def _gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)


def calc_ssim(pred, gt, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    SSIM between two (B, C, H, W) tensors. Returns mean over batch.
    Higher is better.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    C = pred.size(1)
    window = _gaussian_kernel(window_size).to(pred.device)
    window = window.expand(C, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=C)

    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ===================== No-Reference Metrics =====================

def _rgb_to_lab(img_np):
    """Convert RGB [0,1] numpy image (H,W,3) to CIELab (approximate)."""
    # sRGB -> linear
    mask = img_np > 0.04045
    linear = np.where(mask, ((img_np + 0.055) / 1.055) ** 2.4, img_np / 12.92)

    # Linear RGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])

    # XYZ -> Lab
    eps = 0.008856
    kappa = 903.3
    mask = xyz > eps
    f = np.where(mask, xyz ** (1/3), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return L, a, b


def calc_uciqe(img):
    """
    UCIQE: Underwater Color Image Quality Evaluation.
    No-reference metric. Higher is better.
    
    UCIQE = c1·σ_c + c2·con_l + c3·μ_s
    where σ_c = chroma std, con_l = luminance contrast, μ_s = avg saturation.
    
    Args:
        img: (C,H,W) tensor or (H,W,3) numpy, values in [0,1]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    img = np.clip(img, 0, 1).astype(np.float64)

    L, a, b = _rgb_to_lab(img)

    # Chroma
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    # Luminance contrast (top 1% - bottom 1%)
    L_flat = L.flatten()
    con_l = np.percentile(L_flat, 99) - np.percentile(L_flat, 1)

    # Saturation
    hsv_s = chroma / (np.sqrt(L ** 2 + chroma ** 2) + 1e-8)
    mu_s = np.mean(hsv_s)

    # Standard UCIQE weights
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return c1 * sigma_c + c2 * con_l + c3 * mu_s


def calc_uiqm(img):
    """
    UIQM: Underwater Image Quality Measure (simplified).
    UIQM = c1·UICM + c2·UISM + c3·UIConM
    
    Args:
        img: (C,H,W) tensor or (H,W,3) numpy, values in [0,1]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    img = np.clip(img, 0, 1).astype(np.float64)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # UICM — color measure (based on RG, YB chrominance)
    rg = R - G
    yb = 0.5 * (R + G) - B
    mu_rg, mu_yb = np.mean(rg), np.mean(yb)
    sig_rg, sig_yb = np.std(rg), np.std(yb)
    uicm = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sig_rg**2 + sig_yb**2)

    # UISM — sharpness measure (Sobel-based EME)
    from scipy.ndimage import sobel
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    edge = np.sqrt(sx**2 + sy**2)
    uism = np.mean(edge) * 10  # simplified

    # UIConM — contrast measure
    L, _, _ = _rgb_to_lab(img)
    L_norm = L / 100.0
    # Local contrast via block-based approach (simplified)
    k = 8
    h, w = L_norm.shape
    nh, nw = h // k, w // k
    contrast_sum = 0
    count = 0
    for i in range(nh):
        for j in range(nw):
            block = L_norm[i*k:(i+1)*k, j*k:(j+1)*k]
            if block.size > 0:
                bmax, bmin = block.max(), block.min()
                if bmax + bmin > 1e-8:
                    contrast_sum += (bmax - bmin) / (bmax + bmin)
                count += 1
    uiconm = contrast_sum / max(count, 1)

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * uicm + c2 * uism + c3 * uiconm


class MetricTracker:
    """Accumulates and averages metrics over a dataset."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.psnr_vals = []
        self.ssim_vals = []
        self.uciqe_vals = []
        self.uiqm_vals = []

    def update(self, pred, gt=None):
        """pred, gt: (C,H,W) tensors in [0,1]."""
        # 如果有真值图，计算全参考指标
        if gt is not None:
            self.psnr_vals.append(calc_psnr(pred, gt))
            self.ssim_vals.append(calc_ssim(pred.unsqueeze(0), gt.unsqueeze(0)))
        
        # 无论有没有真值图，都计算无参考指标（视觉感知指标）
        self.uciqe_vals.append(calc_uciqe(pred))
        self.uiqm_vals.append(calc_uiqm(pred))

    def summary(self):
        res = {}
        # 只有在 psnr_vals 不为空时才计算平均值，防止报错
        if self.psnr_vals:
            res['PSNR'] = np.mean(self.psnr_vals)
            res['SSIM'] = np.mean(self.ssim_vals)
        res['UCIQE'] = np.mean(self.uciqe_vals)
        res['UIQM'] = np.mean(self.uiqm_vals)
        return res