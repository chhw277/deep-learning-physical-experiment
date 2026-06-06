"""Stokes z-score 统计量（可选；当前训练/推理使用物理尺度，不调用本模块）。"""

import numpy as np

# 各通道 mean 和 std，顺序: S0, S1, S2, S3
MEAN = np.array([0.2026, 0.0257, -0.0228, 0.0077], dtype=np.float32)
STD = np.array([0.3158, 0.3167, 0.2863, 0.1789], dtype=np.float32)


def normalize(stokes: np.ndarray) -> np.ndarray:
    """(H,W,4) 原始Stokes → 归一化到均值0方差1。"""
    return (stokes - MEAN) / STD


def denormalize(stokes: np.ndarray) -> np.ndarray:
    """归一化Stokes → 原始尺度。"""
    return stokes * STD + MEAN
