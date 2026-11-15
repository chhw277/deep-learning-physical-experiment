"""Legacy metric API with explicit range and per-image batch SSIM."""
import numpy as np
from skimage.metrics import structural_similarity


def _arrays(pred, gt, data_range):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim not in (3, 4) or pred.shape[-1] != 4:
        raise ValueError('Expected matching (H,W,4) or (N,H,W,4) arrays')
    if not pred.size or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('Expected non-empty finite arrays')
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError('data_range must be positive and finite')
    if pred.ndim == 3:
        pred, gt = pred[None], gt[None]
    return pred, gt


def psnr_channels(pred, gt, data_range=1.0):
    """Four channel PSNR from pooled MSE; 99 marks near-zero MSE."""
    pred, gt = _arrays(pred, gt, data_range)
    mse = np.mean((pred - gt) ** 2, axis=(0, 1, 2))
    return np.array([99.0 if v < 1e-12 else 10 * np.log10(data_range**2 / v) for v in mse])


def mean_ssim(pred, gt, data_range=1.0):
    pred, gt = _arrays(pred, gt, data_range)
    if min(pred.shape[1:3]) < 7:
        raise ValueError('SSIM requires height and width >= 7')
    return float(np.mean([structural_similarity(g[..., c], p[..., c], data_range=data_range)
                          for p, g in zip(pred, gt) for c in range(4)]))
