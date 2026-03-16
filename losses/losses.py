"""
Loss functions for Phy-FreqFusion.

L_total = L_rec + λ1·L_phy + λ2·L_freq + λ3·L_edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L1 pixel-level reconstruction loss (Eq. 6)."""
    def forward(self, J_pred, I_gt):
        return F.l1_loss(J_pred, I_gt)


class WidebandPhysicsLoss(nn.Module):
    """
    Self-supervised physics-consistency loss (Eq. 7).
    
    Re-compose degraded image from predicted outputs via the optical model
    and enforce it to match the original input:
        I_recon = J_pred ⊙ t_c + B_∞ ⊙ (1 - t_c)
        L_phy = ||I - I_recon||₁
    """
    def forward(self, I, J_pred, t_c, B_inf):
        I_recon = J_pred * t_c + B_inf * (1.0 - t_c)
        return F.l1_loss(I_recon, I)


class FrequencyLoss(nn.Module):
    """
    Frequency-domain consistency loss (Eq. 8).
    
    L_freq = ||F(I_out) - F(I_gt)||₁
    """
    def forward(self, J_pred, I_gt):
        fft_pred = torch.fft.rfft2(J_pred, norm='ortho')
        fft_gt = torch.fft.rfft2(I_gt, norm='ortho')
        return F.l1_loss(fft_pred.abs(), fft_gt.abs()) + \
               F.l1_loss(fft_pred.angle(), fft_gt.angle())


class EdgePreservationLoss(nn.Module):
    """
    Laplacian-based edge preservation loss (Eq. 9).
    
    L_edge = ||Δ(I_out) - Δ(I_gt)||₁
    where Δ is the discrete Laplacian operator.
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        self.register_buffer('kernel', kernel)

    def _laplacian(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        lap = F.conv2d(x, self.kernel.to(x.device), padding=1)
        return lap.reshape(B, C, H, W)

    def forward(self, J_pred, I_gt):
        lap_pred = self._laplacian(J_pred)
        lap_gt = self._laplacian(I_gt)
        return F.l1_loss(lap_pred, lap_gt)


class PhyFreqFusionLoss(nn.Module):
    """
    Combined total loss (Eq. 10):
        L_total = L_rec + λ1·L_phy + λ2·L_freq + λ3·L_edge
    """
    def __init__(self, lambda_phy=0.5, lambda_freq=0.1, lambda_edge=0.2):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.lambda_freq = lambda_freq
        self.lambda_edge = lambda_edge

        self.rec_loss = ReconstructionLoss()
        self.phy_loss = WidebandPhysicsLoss()
        self.freq_loss = FrequencyLoss()
        self.edge_loss = EdgePreservationLoss()

    def forward(self, model_output, I_input, I_gt):
        """
        Args:
            model_output: dict from PhyFreqFusion.forward()
            I_input:      original degraded image (B, 3, H, W)
            I_gt:         ground truth clean image (B, 3, H, W)
        Returns:
            total_loss, loss_dict
        """
        J_pred = model_output['J_pred']
        t_c = model_output['t_c']
        B_inf = model_output['B_inf']

        l_rec = self.rec_loss(J_pred, I_gt)
        l_phy = self.phy_loss(I_input, J_pred, t_c, B_inf)
        l_freq = self.freq_loss(J_pred, I_gt)
        l_edge = self.edge_loss(J_pred, I_gt)

        total = l_rec + self.lambda_phy * l_phy + \
                self.lambda_freq * l_freq + self.lambda_edge * l_edge

        loss_dict = {
            'total': total.item(),
            'rec': l_rec.item(),
            'phy': l_phy.item(),
            'freq': l_freq.item(),
            'edge': l_edge.item(),
        }
        return total, loss_dict
