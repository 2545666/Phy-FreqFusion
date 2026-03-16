"""
Default configuration for Phy-FreqFusion experiments.

This file documents all hyperparameters referenced in the paper.
Import and modify as needed, or use train.py CLI args directly.
"""


class Config:
    """
    Phy-FreqFusion default configuration.
    
    Follows the settings described in Section IV-A:
      Implementation Details.
    """

    # ---- Model Architecture ----
    base_ch = 64          # U-Net base channels (encoder: 64→128→256→512→512)
    lap_levels = 3        # Laplacian pyramid levels in spatial stream
    eps = 1e-6            # Physical inversion stability constant ε

    # ---- Training ----
    epochs = 200
    batch_size = 8        # Per-GPU batch size
    img_size = 256        # Training crop size (H=W)
    lr = 2e-4             # Initial learning rate (AdamW)
    weight_decay = 1e-4   # AdamW weight decay
    lr_scheduler = 'cosine'  # CosineAnnealingLR
    lr_min = 1e-6         # Minimum LR at end of cosine schedule
    grad_clip = 5.0       # Max gradient norm

    # ---- Loss Weights (Eq. 10) ----
    lambda_phy = 0.5      # λ₁: wideband physics loss weight
    lambda_freq = 0.1     # λ₂: frequency domain loss weight
    lambda_edge = 0.2     # λ₃: edge preservation loss weight

    # ---- Data Augmentation ----
    augment_hflip = True   # Random horizontal flip
    augment_vflip = True   # Random vertical flip
    augment_crop = True    # Random crop (from img_size+32 → img_size)

    # ---- Datasets ----
    # UIEB: 890 pairs total, 800 train / 90 test
    uieb_train_split = 800
    # EUVP: uses trainA/trainB and validation splits

    # ---- Evaluation Metrics ----
    # Full-reference: PSNR (dB, ↑), SSIM (↑)
    # No-reference:   UCIQE (↑), UIQM (↑)

    # ---- Miscellaneous ----
    num_workers = 4
    pin_memory = True
    mixed_precision = True    # Use AMP (FP16)
    save_interval = 10        # Checkpoint every N epochs
    log_interval = 50         # Print loss every N steps
    seed = 42


# ---- Ablation Variant Configs ----

class ConfigNoPhysics(Config):
    """w/o Physics Sub-net ablation."""
    variant = 'no_physics'
    lambda_phy = 0.0  # No physics loss when physics module removed


class ConfigNoFFT(Config):
    """w/o Frequency Stream ablation."""
    variant = 'no_fft'
    lambda_freq = 0.0  # No frequency loss when FFT stream removed


class ConfigNoSNRGate(Config):
    """w/o SNR Gating ablation."""
    variant = 'no_snr_gate'
    # All losses remain active; only gating mechanism changes


# ---- Recommended Settings per Dataset ----

class ConfigUIEB(Config):
    """Optimized for UIEB benchmark."""
    dataset = 'uieb'
    epochs = 200
    batch_size = 8
    img_size = 256


class ConfigEUVP(Config):
    """Optimized for EUVP benchmark."""
    dataset = 'euvp'
    epochs = 150        # EUVP is larger, fewer epochs needed
    batch_size = 8
    img_size = 256
