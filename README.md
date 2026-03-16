# Phy-FreqFusion

**Wideband Physics-Guided and SNR-Aware Spatial-Frequency Interactive Network for Underwater Image Enhancement**

> IEEE Conference Paper Implementation

---

## Architecture Overview

```
Input I(x)
    │
    ▼
┌─────────────────────────────────┐
│  Stage I: Wideband Physics-     │
│  Guided Sub-net                 │
│                                 │
│  U-Net Backbone                 │
│   ├─ Head A → t_c (3-ch trans.) │  ← Wideband: R/G/B independent
│   └─ Head B → B_∞ (background)  │  ← GAP for global receptive field
│                                 │
│  Physical Inversion Layer:      │
│  J_coarse = (I-B∞)/max(tc,ε)+B∞│
└────────────┬────────────────────┘
             │ J_coarse, t_c
             ▼
┌─────────────────────────────────┐
│  Stage II: SNR-Aware Spatial-   │
│  Frequency Fusion               │
│                                 │
│  ┌─────────┐   ┌─────────────┐ │
│  │Freq.    │   │Spatial      │ │
│  │Stream   │   │Stream       │ │
│  │ FFT →   │   │ Laplacian   │ │
│  │ AmpNet →│   │ Pyramid →   │ │
│  │ IFFT    │   │ Refinement  │ │
│  │→ F_freq │   │→ F_spatial  │ │
│  └────┬────┘   └─────┬───────┘ │
│       │               │        │
│       └───┬───────┬───┘        │
│           ▼       ▼            │
│    SNR-Aware Confidence Gate:  │
│    F = tc·F_sp + (1-tc)·F_fq  │
│           │                    │
│           ▼                    │
│    Refinement Conv → J_pred    │
└─────────────────────────────────┘
```

## Project Structure

```
phy_freqfusion/
├── models/
│   └── phy_freqfusion.py    # Full architecture + ablation variants
├── datasets/
│   └── uie_dataset.py       # UIEB / EUVP dataloaders
├── losses/
│   └── losses.py            # L_rec, L_phy, L_freq, L_edge
├── utils/
│   └── metrics.py           # PSNR, SSIM, UCIQE, UIQM
├── train.py                 # Training pipeline
├── test.py                  # Evaluation & inference
├── run_ablation.sh          # Ablation study runner
└── README.md
```

## Dataset Preparation

### UIEB (890 pairs)
```
data/UIEB/
├── raw/           # 890 degraded images
└── reference/     # 890 reference images (matching filenames)
```
Download: https://li-chongyi.github.io/proj_benchmark.html

### EUVP
```
data/EUVP/underwater_scenes/
├── trainA/        # degraded
├── trainB/        # reference
└── validation/
    ├── input/
    └── target/
```
Download: https://irvlab.cs.umn.edu/resources/euvp-dataset

## Training

```bash
# Full model on UIEB
python train.py \
    --data_root ./data/UIEB \
    --dataset uieb \
    --epochs 200 \
    --batch_size 8 \
    --lr 2e-4 \
    --amp

# With custom loss weights
python train.py \
    --data_root ./data/UIEB \
    --lambda_phy 0.5 \
    --lambda_freq 0.1 \
    --lambda_edge 0.2 \
    --amp
```

## Evaluation

```bash
# Full evaluation with metrics
python test.py \
    --data_root ./data/UIEB \
    --dataset uieb \
    --checkpoint ./checkpoints/best.pth \
    --save_images \
    --output_dir ./results

# Inference only (no GT)
python test.py \
    --mode infer \
    --input_dir ./my_underwater_images \
    --checkpoint ./checkpoints/best.pth
```

## Ablation Study

```bash
# Run all 4 variants
bash run_ablation.sh
```

| Variant           | Description                         |
|-------------------|-------------------------------------|
| `full`            | Complete Phy-FreqFusion             |
| `no_physics`      | w/o Stage I physics sub-net         |
| `no_fft`          | w/o frequency stream (spatial only) |
| `no_snr_gate`     | w/o SNR-aware gating (avg fusion)   |

## Loss Functions

| Loss     | Formula                                          | Purpose                        |
|----------|--------------------------------------------------|--------------------------------|
| L_rec    | ‖J_pred − I_gt‖₁                                | Pixel fidelity                 |
| L_phy    | ‖I − (J_pred·t_c + B∞·(1−t_c))‖₁              | Self-supervised physics        |
| L_freq   | ‖FFT(J_pred) − FFT(I_gt)‖₁                     | Frequency consistency          |
| L_edge   | ‖Δ(J_pred) − Δ(I_gt)‖₁                         | Edge/texture preservation      |

**Total:** L = L_rec + 0.5·L_phy + 0.1·L_freq + 0.2·L_edge

## Key Implementation Details

- **Wideband Transmission**: 3-channel (R/G/B independent) instead of mono-channel
- **Physical Inversion**: Deterministic layer, no learnable params — pure formula
- **Amp-Net**: Only modulates FFT amplitude; phase preserved for structure
- **SNR Gating**: t_c acts as confidence — low t_c suppresses spatial sharpening
- **Optimizer**: AdamW + CosineAnnealing LR schedule
- **Mixed Precision**: Supported via `--amp` flag

## Requirements

```
torch >= 1.12
torchvision
numpy
Pillow
scipy  (for UIQM metric)
```
