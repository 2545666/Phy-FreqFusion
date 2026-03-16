#!/bin/bash
# ============================================================================
# Ablation Study Runner for Phy-FreqFusion
# Trains all four variants and evaluates them.
# ============================================================================

DATA_ROOT="./data/UIEB"
EPOCHS=200
BS=8
LR=2e-4

echo "============================================"
echo "  Phy-FreqFusion Ablation Study"
echo "============================================"

# 1) Full model
echo -e "\n[1/4] Training FULL model..."
python train.py --data_root $DATA_ROOT --variant full \
    --epochs $EPOCHS --batch_size $BS --lr $LR --amp

# 2) w/o Physics Sub-net
echo -e "\n[2/4] Training w/o Physics..."
python train.py --data_root $DATA_ROOT --variant no_physics \
    --epochs $EPOCHS --batch_size $BS --lr $LR --amp

# 3) w/o FFT (Frequency Stream)
echo -e "\n[3/4] Training w/o FFT..."
python train.py --data_root $DATA_ROOT --variant no_fft \
    --epochs $EPOCHS --batch_size $BS --lr $LR --amp

# 4) w/o SNR Gating
echo -e "\n[4/4] Training w/o SNR Gate..."
python train.py --data_root $DATA_ROOT --variant no_snr_gate \
    --epochs $EPOCHS --batch_size $BS --lr $LR --amp

echo -e "\n============================================"
echo "  All ablation runs complete."
echo "  Run test.py on each checkpoint for metrics."
echo "============================================"
