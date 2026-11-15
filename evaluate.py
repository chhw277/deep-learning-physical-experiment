"""Evaluate paired (H,W,4) Stokes arrays; adapted from the GitHub evaluation scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
from polarized_csi.experiment import describe_file
from polarized_csi.io import stokes_physics


def evaluate(pred: np.ndarray, true: np.ndarray, data_range: str = "target") -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 4:
        raise ValueError(f"Expected matching (H,W,4) arrays, got {pred.shape} and {true.shape}")
    if min(pred.shape[:2]) < 7:
        raise ValueError("SSIM requires height and width >= 7")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Arrays must contain only finite values")
    fixed_range = None if data_range == "target" else float(data_range)
    if fixed_range is not None and (not np.isfinite(fixed_range) or fixed_range <= 0):
        raise ValueError("data_range must be 'target' or a positive finite number")
    channels = []
    for c, name in enumerate(["S0", "S1", "S2", "S3"]):
        p, g = pred[..., c], true[..., c]
        span = float(np.ptp(g))
        dr = fixed_range if fixed_range is not None else (span if span > 1e-10 else 1.0)
        mse = float(np.mean((p - g) ** 2))
        psnr = 99.0 if mse < 1e-12 else float(10 * np.log10(dr**2 / mse))
        ssim = float(structural_similarity(g, p, data_range=dr))
        channels.append(dict(channel=name, data_range=dr, mse=mse, psnr_db=psnr, ssim=ssim))
    return dict(shape=list(pred.shape), range_mode=data_range,
                ground_truth_physics=stokes_physics(true),
                prediction_physics=stokes_physics(pred),
                near_zero_mse_psnr_sentinel_db=99.0, channels=channels,
                mean_psnr_db=float(np.mean([c['psnr_db'] for c in channels])),
                mean_ssim=float(np.mean([c['ssim'] for c in channels])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred', type=Path, required=True, help='Prediction .npy, shape (H,W,4)')
    ap.add_argument('--true', type=Path, required=True, help='Corresponding ground-truth .npy')
    ap.add_argument('--data-range', default='target', help="target (per-channel max-min) or a positive number")
    ap.add_argument('--out-json', type=Path)
    args = ap.parse_args()
    result = evaluate(np.load(args.pred, allow_pickle=False), np.load(args.true, allow_pickle=False), args.data_range)
    result.update(prediction=str(args.pred), ground_truth=str(args.true))
    result['sources'] = {'prediction': describe_file(args.pred), 'ground_truth': describe_file(args.true)}
    print('Channel  data_range  PSNR(dB)  SSIM')
    for c in result['channels']:
        print(f"{c['channel']:<8} {c['data_range']:>10.4f} {c['psnr_db']:>9.2f} {c['ssim']:>7.4f}")
    print(f"Mean PSNR: {result['mean_psnr_db']:.2f} dB; mean SSIM: {result['mean_ssim']:.4f}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
