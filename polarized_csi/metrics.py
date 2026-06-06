"""PSNR / SSIM（物理尺度 Stokes，各通道 data_range 取自真值）。"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def channel_data_range(gt_channel: np.ndarray) -> float:
    """真值通道的动态范围 max - min；近乎常数时回退为 1.0。"""
    dr = float(np.max(gt_channel) - np.min(gt_channel))
    return dr if dr > 1e-10 else 1.0


def psnr_channels(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    pred, gt: (N, H, W, 4) 或 (H, W, 4)，物理尺度 Stokes。
    各通道 PSNR 的 data_range 取该通道真值的 max - min。
    返回长度为 4 的各通道 PSNR (dB)。
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    if pred.ndim == 3:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]
    out = []
    for c in range(4):
        gtc = gt[..., c]
        dr = channel_data_range(gtc)
        mse = np.mean((pred[..., c] - gtc) ** 2)
        if mse < 1e-12:
            out.append(99.0)
        else:
            out.append(10.0 * np.log10((dr**2) / mse))
    return np.array(out, dtype=np.float64)


def mean_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """对四通道取平均 SSIM；各通道 data_range 取自真值。"""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    vals = []
    for c in range(4):
        gtc = gt[..., c]
        dr = channel_data_range(gtc)
        vals.append(
            ssim_fn(
                gtc,
                pred[..., c],
                data_range=dr,
            )
        )
    return float(np.mean(vals))
