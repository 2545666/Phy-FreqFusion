"""
Phy-FreqFusion: Wideband Physics-Guided and SNR-Aware 
Spatial-Frequency Interactive Network for Underwater Image Enhancement

Full architecture implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Conv-BN-LeakyReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBlocks (standard U-Net building block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """MaxPool -> DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample -> Concat skip -> DoubleConv."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes mismatch
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# =============================================================================
# Stage I: Wideband Physics-Guided Sub-net
# =============================================================================

class WidebandPhysicsSubnet(nn.Module):
    """
    U-Net backbone with two disentangled prediction heads:
      - Head A: Channel-wise transmission map  t_c ∈ R^{B×3×H×W}
      - Head B: Global background light        B_∞ ∈ R^{B×3×1×1}
    
    Then applies a deterministic Physical Inversion Layer.
    """

    def __init__(self, base_ch=64, eps=1e-6):
        super().__init__()
        self.eps = eps

        # ---- U-Net Encoder ----
        self.enc1 = DoubleConv(3, base_ch)           # H×W
        self.enc2 = DownBlock(base_ch, base_ch * 2)   # H/2
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)  # H/4
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)  # H/8

        # ---- Bottleneck ----
        self.bottleneck = DownBlock(base_ch * 8, base_ch * 8)  # H/16

        # ---- U-Net Decoder (shared) ----
        self.dec4 = UpBlock(base_ch * 16, base_ch * 4)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 2)
        self.dec2 = UpBlock(base_ch * 4, base_ch)
        self.dec1 = UpBlock(base_ch * 2, base_ch)

        # ---- Head A: Transmission Map (channel-wise, 3-ch) ----
        self.head_trans = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch // 2, 3, 1),
            nn.Sigmoid(),  # transmission ∈ (0, 1)
        )

        # ---- Head B: Background Light (global, 3-ch) ----
        # Uses GAP on bottleneck features to enforce global receptive field
        self.head_bg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # B×C×1×1
            nn.Flatten(),                            # B×C
            nn.Linear(base_ch * 8, base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_ch * 2, 3),
            nn.Sigmoid(),                            # B_∞ ∈ (0, 1)
        )

    def forward(self, I):
        """
        Args:
            I: degraded image, (B, 3, H, W)
        Returns:
            J_coarse: coarse restored image
            t_c:      channel-wise transmission map (B, 3, H, W)
            B_inf:    global background light       (B, 3, 1, 1)
        """
        # Encoder
        e1 = self.enc1(I)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Head A — Transmission Map
        t_c = self.head_trans(d1)  # (B, 3, H, W)

        # Head B — Background Light (from bottleneck features)
        B_inf = self.head_bg(bn)   # (B, 3)
        B_inf = B_inf.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)

        # Physical Inversion Layer (Eq. 2)
        #   J_coarse = (I - B_∞) / max(t_c, ε) + B_∞
        J_coarse = (I - B_inf) / torch.clamp(t_c, min=self.eps) + B_inf
        J_coarse = torch.clamp(J_coarse, 0.0, 1.0)

        return J_coarse, t_c, B_inf


# =============================================================================
# Stage II: SNR-Aware Spatial-Frequency Fusion
# =============================================================================

class AmpNet(nn.Module):
    """
    Lightweight convolutional network that modulates ONLY the amplitude
    spectrum in Fourier domain for global color correction.
    """
    def __init__(self, ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, ch, 3, 1, 1),
            nn.Softplus(),  # ensure positive amplitude scaling
        )

    def forward(self, amplitude):
        return self.net(amplitude)


class FrequencyStream(nn.Module):
    """
    Frequency-domain color correction via FFT amplitude modulation.
    
    Steps:
      1) 2D FFT  →  Amplitude + Phase
      2) Amp-Net modulates amplitude
      3) 2D IFFT  →  corrected spatial features
    """
    def __init__(self):
        super().__init__()
        self.amp_net = AmpNet(ch=3)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — J_coarse
        Returns:
            F_freq: (B, 3, H, W) — color-corrected features
        """
        # 2D FFT (complex)
        fft = torch.fft.rfft2(x, norm='ortho')
        amplitude = fft.abs()   # (B, 3, H, W//2+1)
        phase = fft.angle()     # (B, 3, H, W//2+1)

        # Modulate amplitude
        amp_mod = self.amp_net(amplitude)

        # Reconstruct with modulated amplitude + original phase
        fft_mod = amp_mod * torch.exp(1j * phase)
        F_freq = torch.fft.irfft2(fft_mod, s=x.shape[-2:], norm='ortho')
        return F_freq


class LaplacianPyramid(nn.Module):
    """
    Extracts multi-scale high-frequency features via Laplacian Pyramid.
    Operates on 3-channel images; outputs refined high-frequency map.
    """
    def __init__(self, n_levels=3):
        super().__init__()
        self.n_levels = n_levels

        # Learnable refinement convolutions for each pyramid level
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1),
            )
            for _ in range(n_levels)
        ])

        # Final aggregation
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * n_levels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    @staticmethod
    def _gaussian_downsample(x):
        """Simple Gaussian-like down by factor 2 using avg pool."""
        return F.avg_pool2d(x, 2)

    @staticmethod
    def _upsample(x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            F_spatial: (B, 3, H, W) — enhanced high-frequency features
        """
        H, W = x.shape[2], x.shape[3]
        laplacians = []
        current = x
        for i in range(self.n_levels):
            down = self._gaussian_downsample(current)
            up = self._upsample(down, (current.shape[2], current.shape[3]))
            lap = current - up  # high-frequency residual
            lap_refined = self.refine[i](lap)
            # Resize all to original resolution for fusion
            lap_refined = self._upsample(lap_refined, (H, W)) if lap_refined.shape[2:] != (H, W) else lap_refined
            laplacians.append(lap_refined)
            current = down

        # Fuse multi-scale high-freq features
        F_spatial = self.fuse(torch.cat(laplacians, dim=1))
        return F_spatial


class SpatialFrequencyFusion(nn.Module):
    """
    Stage II: SNR-Aware Spatial-Frequency Fusion.
    
    Dual-stream:
      - Frequency stream: global color correction via FFT
      - Spatial stream:   local detail enhancement via Laplacian pyramid
    
    SNR-Aware Gating:
      F_out = t_c ⊙ F_spatial + (1 - t_c) ⊙ F_freq
      
    A final refinement block produces the enhanced output.
    """
    def __init__(self, lap_levels=3):
        super().__init__()
        self.freq_stream = FrequencyStream()
        self.spatial_stream = LaplacianPyramid(n_levels=lap_levels)

        # Final refinement convolutions
        self.refine = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),  # output ∈ [0, 1]
        )

    def forward(self, J_coarse, t_c):
        """
        Args:
            J_coarse: (B, 3, H, W) coarse restored image from Stage I
            t_c:      (B, 3, H, W) channel-wise transmission map
        Returns:
            J_pred: (B, 3, H, W) final enhanced image
        """
        F_freq = self.freq_stream(J_coarse)
        F_spatial = self.spatial_stream(J_coarse)

        # SNR-Aware Confidence Gating (Eq. 5)
        F_out = t_c * F_spatial + (1.0 - t_c) * F_freq

        # Residual refinement: add coarse image as skip
        J_pred = self.refine(J_coarse + F_out)
        return J_pred


# =============================================================================
# Full Model: Phy-FreqFusion
# =============================================================================

class PhyFreqFusion(nn.Module):
    """
    Phy-FreqFusion: Two-stage cascaded end-to-end network.
    
    Stage I:  Wideband Physics-Guided Sub-net
    Stage II: SNR-Aware Spatial-Frequency Fusion
    """
    def __init__(self, base_ch=64, lap_levels=3, eps=1e-6):
        super().__init__()
        self.stage1 = WidebandPhysicsSubnet(base_ch=base_ch, eps=eps)
        self.stage2 = SpatialFrequencyFusion(lap_levels=lap_levels)

    def forward(self, I):
        """
        Args:
            I: degraded underwater image (B, 3, H, W), values in [0, 1]
        Returns:
            dict with keys:
                'J_pred':    final enhanced image
                'J_coarse':  coarse restored image (Stage I output)
                't_c':       transmission map
                'B_inf':     background light
        """
        # Stage I — Physical inversion
        J_coarse, t_c, B_inf = self.stage1(I)

        # Stage II — Spatial-frequency refinement
        J_pred = self.stage2(J_coarse, t_c)

        return {
            'J_pred': J_pred,
            'J_coarse': J_coarse,
            't_c': t_c,
            'B_inf': B_inf,
        }


# =============================================================================
# Ablation Variants
# =============================================================================

class PhyFreqFusion_NoPhysics(nn.Module):
    """Ablation: w/o Physics Sub-net — direct data-driven."""
    def __init__(self, base_ch=64, lap_levels=3):
        super().__init__()
        self.stage2 = SpatialFrequencyFusion(lap_levels=lap_levels)
        # A simple dummy transmission (uniform 0.5) and skip Stage I
        self.direct_encoder = nn.Sequential(
            ConvBlock(3, base_ch),
            ConvBlock(base_ch, 3),
        )

    def forward(self, I):
        J_coarse = self.direct_encoder(I)
        t_c = torch.full_like(I, 0.5)
        J_pred = self.stage2(J_coarse, t_c)
        return {'J_pred': J_pred, 'J_coarse': J_coarse,
                't_c': t_c, 'B_inf': torch.zeros(I.size(0), 3, 1, 1, device=I.device)}


class PhyFreqFusion_NoFFT(nn.Module):
    """Ablation: w/o Frequency Stream — spatial-only Stage II."""
    def __init__(self, base_ch=64, lap_levels=3, eps=1e-6):
        super().__init__()
        self.stage1 = WidebandPhysicsSubnet(base_ch=base_ch, eps=eps)
        self.spatial_stream = LaplacianPyramid(n_levels=lap_levels)
        self.refine = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, I):
        J_coarse, t_c, B_inf = self.stage1(I)
        F_spatial = self.spatial_stream(J_coarse)
        J_pred = self.refine(J_coarse + F_spatial)
        return {'J_pred': J_pred, 'J_coarse': J_coarse, 't_c': t_c, 'B_inf': B_inf}


class PhyFreqFusion_NoSNRGate(nn.Module):
    """Ablation: w/o SNR Gating — simple averaging of dual streams."""
    def __init__(self, base_ch=64, lap_levels=3, eps=1e-6):
        super().__init__()
        self.stage1 = WidebandPhysicsSubnet(base_ch=base_ch, eps=eps)
        self.freq_stream = FrequencyStream()
        self.spatial_stream = LaplacianPyramid(n_levels=lap_levels)
        self.refine = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, I):
        J_coarse, t_c, B_inf = self.stage1(I)
        F_freq = self.freq_stream(J_coarse)
        F_spatial = self.spatial_stream(J_coarse)
        # Simple average instead of SNR gating
        F_out = 0.5 * F_spatial + 0.5 * F_freq
        J_pred = self.refine(J_coarse + F_out)
        return {'J_pred': J_pred, 'J_coarse': J_coarse, 't_c': t_c, 'B_inf': B_inf}


if __name__ == '__main__':
    # Quick sanity check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhyFreqFusion(base_ch=64).to(device)
    x = torch.randn(2, 3, 256, 256).clamp(0, 1).to(device)
    out = model(x)
    for k, v in out.items():
        print(f"{k}: {v.shape}")
    
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal parameters: {n_params:.2f} M")
