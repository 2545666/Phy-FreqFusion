#!/usr/bin/env bash
# =============================================================================
#  run_sensitivity.sh — Hyperparameter Sensitivity Analysis
#
#  Performs a grid search over:
#    λ₂ (lambda_phy)  ∈ {0.1, 0.5, 1.0}
#    λ₃ (lambda_freq) ∈ {0.05, 0.1, 0.5}
#
#  Fixed:
#    λ₁ (reconstruction) = 1.0  (implicitly 1.0 in PhyFreqFusionLoss)
#    λ₄ (lambda_edge)    = 0.2
#
#  Each (λ₂, λ₃) combination trains a full model on UIEB, validates on
#  the 90-image test split, and logs the best PSNR/SSIM to a shared CSV.
#
#  Usage:
#    chmod +x run_sensitivity.sh
#    bash run_sensitivity.sh
#
#  After completion, run:
#    python plot_sensitivity.py --results_csv sensitivity_results.csv
# =============================================================================

set -euo pipefail

# ---- User-configurable paths ----
DATA_ROOT="./data/UIEB"             # UIEB dataset root
SAVE_ROOT="./checkpoints/sensitivity"
RESULTS_CSV="./sensitivity_results.csv"
EPOCHS=200                           # Full training (reduce for debugging)
BATCH_SIZE=8
IMG_SIZE=256
LAMBDA_EDGE=0.2                      # Fixed λ₄

# Grid values
LAMBDA_PHY_GRID=(0.1 0.5 1.0)
LAMBDA_FREQ_GRID=(0.05 0.1 0.5)

# ---- Write CSV header ----
echo "lambda_phy,lambda_freq,best_psnr,best_ssim" > "${RESULTS_CSV}"

echo "============================================================"
echo "  Phy-FreqFusion Hyperparameter Sensitivity Analysis"
echo "  Grid: lambda_phy=${LAMBDA_PHY_GRID[*]}"
echo "        lambda_freq=${LAMBDA_FREQ_GRID[*]}"
echo "  Fixed: lambda_edge=${LAMBDA_EDGE}"
echo "  Epochs: ${EPOCHS}  |  Batch: ${BATCH_SIZE}"
echo "============================================================"
echo ""

RUN_IDX=0
TOTAL=$(( ${#LAMBDA_PHY_GRID[@]} * ${#LAMBDA_FREQ_GRID[@]} ))

for LPHY in "${LAMBDA_PHY_GRID[@]}"; do
    for LFREQ in "${LAMBDA_FREQ_GRID[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        TAG="lphy${LPHY}_lfreq${LFREQ}"
        SAVE_DIR="${SAVE_ROOT}/${TAG}"

        echo "------------------------------------------------------------"
        echo "  Run ${RUN_IDX}/${TOTAL}: lambda_phy=${LPHY}, lambda_freq=${LFREQ}"
        echo "  Save dir: ${SAVE_DIR}"
        echo "------------------------------------------------------------"

        # Run training
        python train.py \
            --data_root "${DATA_ROOT}" \
            --dataset uieb \
            --img_size ${IMG_SIZE} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr 2e-4 \
            --weight_decay 1e-4 \
            --lambda_phy "${LPHY}" \
            --lambda_freq "${LFREQ}" \
            --lambda_edge "${LAMBDA_EDGE}" \
            --variant full \
            --amp \
            --save_dir "${SAVE_DIR}" \
            --save_interval 10 \
            --log_interval 50

        # ---- Extract best metrics from the saved checkpoint ----
        # The training script saves best.pth with val_psnr and val_ssim.
        # We use a tiny Python one-liner to read them.
        BEST_CKPT="${SAVE_DIR}/$(ls -t ${SAVE_DIR}/ | head -1)/best.pth"

        # Fallback: find best.pth anywhere under SAVE_DIR
        if [ ! -f "${BEST_CKPT}" ]; then
            BEST_CKPT=$(find "${SAVE_DIR}" -name "best.pth" | head -1)
        fi

        if [ -f "${BEST_CKPT}" ]; then
            METRICS=$(python -c "
import torch, sys
ckpt = torch.load('${BEST_CKPT}', map_location='cpu', weights_only=False)
psnr = ckpt.get('val_psnr', ckpt.get('best_psnr', 0.0))
ssim = ckpt.get('val_ssim', 0.0)
print(f'{psnr:.4f},{ssim:.6f}')
")
            echo "${LPHY},${LFREQ},${METRICS}" >> "${RESULTS_CSV}"
            echo "  >> Logged: PSNR/SSIM = ${METRICS}"
        else
            echo "  >> WARNING: best.pth not found under ${SAVE_DIR}"
            echo "${LPHY},${LFREQ},0.0,0.0" >> "${RESULTS_CSV}"
        fi

        echo ""
    done
done

echo "============================================================"
echo "  Sensitivity sweep complete!"
echo "  Results saved to: ${RESULTS_CSV}"
echo ""
echo "  Next step:"
echo "    python plot_sensitivity.py --results_csv ${RESULTS_CSV}"
echo "============================================================"
