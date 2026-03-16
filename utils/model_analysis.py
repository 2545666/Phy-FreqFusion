"""
Model analysis: parameter counts, FLOPs estimation, throughput benchmarking.

Usage:
    python -m utils.model_analysis

Outputs a table suitable for the paper's implementation details section.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.phy_freqfusion import (
    PhyFreqFusion, PhyFreqFusion_NoPhysics,
    PhyFreqFusion_NoFFT, PhyFreqFusion_NoSNRGate
)


def count_params(module, name=''):
    """Count trainable parameters."""
    total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total


def params_breakdown(model):
    """Per-component parameter breakdown."""
    breakdown = {}

    if hasattr(model, 'stage1'):
        s1 = model.stage1
        # Encoder + decoder
        enc_dec_params = (
            count_params(s1.enc1) + count_params(s1.enc2) +
            count_params(s1.enc3) + count_params(s1.enc4) +
            count_params(s1.bottleneck) +
            count_params(s1.dec4) + count_params(s1.dec3) +
            count_params(s1.dec2) + count_params(s1.dec1)
        )
        breakdown['Stage I: U-Net Backbone'] = enc_dec_params
        breakdown['Stage I: Head A (Transmission)'] = count_params(s1.head_trans)
        breakdown['Stage I: Head B (Background)'] = count_params(s1.head_bg)

    if hasattr(model, 'stage2'):
        s2 = model.stage2
        if hasattr(s2, 'freq_stream'):
            breakdown['Stage II: Frequency Stream (Amp-Net)'] = count_params(s2.freq_stream)
        if hasattr(s2, 'spatial_stream'):
            breakdown['Stage II: Spatial Stream (Lap. Pyramid)'] = count_params(s2.spatial_stream)
        breakdown['Stage II: Refinement'] = count_params(s2.refine)

    breakdown['Total'] = count_params(model)
    return breakdown


def estimate_flops(model, input_size=(1, 3, 256, 256)):
    """
    Rough FLOPs estimation using a forward hook approach.
    For precise numbers, use thop or fvcore.
    """
    try:
        from thop import profile
        x = torch.randn(*input_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except ImportError:
        # Simple estimation: ~2 FLOPs per parameter per pixel
        params = count_params(model)
        h, w = input_size[2], input_size[3]
        estimated_flops = params * h * w * 2  # very rough
        return estimated_flops, params


def benchmark_throughput(model, input_size=(1, 3, 256, 256), device='cpu',
                         warmup=10, n_runs=50):
    """Measure inference throughput."""
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)

    avg_ms = sum(times) / len(times) * 1000
    fps = 1000.0 / avg_ms
    return avg_ms, fps


def print_analysis():
    print('=' * 70)
    print('  Phy-FreqFusion — Model Analysis')
    print('=' * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for base_ch in [32, 64]:
        print(f'\n--- base_ch = {base_ch} ---')
        model = PhyFreqFusion(base_ch=base_ch, lap_levels=3)
        breakdown = params_breakdown(model)

        print(f'\n  Parameter Breakdown:')
        for name, count in breakdown.items():
            print(f'    {name:<45} {count/1e6:>8.3f} M')

        # Throughput
        for res in [256]:
            avg_ms, fps = benchmark_throughput(
                model, input_size=(1, 3, res, res),
                device=device, warmup=5, n_runs=20
            )
            print(f'\n  Throughput @ {res}x{res} ({device}):')
            print(f'    Avg latency: {avg_ms:.1f} ms')
            print(f'    Throughput:  {fps:.1f} FPS')

    # Ablation variant comparison
    print(f'\n--- Ablation Variant Parameter Counts (base_ch=64) ---')
    variants = {
        'Full': PhyFreqFusion(base_ch=64),
        'w/o Physics': PhyFreqFusion_NoPhysics(base_ch=64),
        'w/o FFT': PhyFreqFusion_NoFFT(base_ch=64),
        'w/o SNR Gate': PhyFreqFusion_NoSNRGate(base_ch=64),
    }
    print(f'  {"Variant":<25} {"Params (M)":>12}')
    print(f'  {"-"*40}')
    for name, m in variants.items():
        n = count_params(m) / 1e6
        print(f'  {name:<25} {n:>12.3f}')

    # Generate implementation details paragraph
    model = PhyFreqFusion(base_ch=64, lap_levels=3)
    total_params = count_params(model) / 1e6
    avg_ms, fps = benchmark_throughput(
        model, input_size=(1, 3, 256, 256), device=device, warmup=5, n_runs=20)

    print(f'\n--- Suggested Implementation Details Paragraph ---')
    print(f"""
All experiments are conducted using PyTorch on a single NVIDIA RTX 3090 GPU.
The network is trained for 200 epochs using the AdamW optimizer with an
initial learning rate of 2×10⁻⁴, weight decay of 1×10⁻⁴, and a cosine
annealing schedule decaying to 1×10⁻⁶. The batch size is set to 8 with
input images randomly cropped to 256×256 pixels. Data augmentation includes
random horizontal and vertical flips. The loss weights are set to
λ₁ = 0.5, λ₂ = 0.1, λ₃ = 0.2. The stability constant ε in the physical
inversion layer is set to 1×10⁻⁶. The Laplacian pyramid uses 3 levels.
The full model contains approximately {total_params:.2f}M parameters.
""")


if __name__ == '__main__':
    print_analysis()
