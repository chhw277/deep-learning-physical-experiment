"""Shared radiometric input contract for labels and both ML frameworks."""
from pathlib import Path

import numpy as np
from PIL import Image


def load_measurement(path: str | Path, white_level: float | None = None) -> np.ndarray:
    """Preserve grayscale precision and return [0,1]; NPY/float images are normalized.

    Use the same white level for all captures. Never independently min/max scale
    an image. Supply white_level for 10/12-bit cameras in a 16-bit container.
    """
    path = Path(path)
    if white_level is not None and (not np.isfinite(white_level) or white_level <= 0):
        raise ValueError('white_level must be positive and finite')
    if path.suffix.lower() == '.npy':
        raw = np.load(path, allow_pickle=False)
        scale = 1.0 if white_level is None else white_level
    else:
        with Image.open(path) as image:
            mode, fmt, raw = image.mode, image.format, np.array(image)
        if raw.ndim != 2 or raw.dtype.kind not in 'uif':
            raise ValueError(f'Expected grayscale intensity: {path} ({mode})')
        if white_level is not None:
            scale = white_level
        elif raw.dtype == np.uint8:
            scale = 255.0
        elif raw.dtype == np.uint16 or mode.startswith('I;16'):
            scale = 65535.0
        elif mode == 'I' and fmt == 'PNG':
            scale = 65535.0  # Pillow can decode 16-bit PNG into an int32 container.
        elif raw.dtype.kind == 'f':
            scale = 1.0
        else:
            raise ValueError(f'Unknown sensor range: {path}; specify white_level')
    if raw.ndim != 2 or not raw.size or raw.dtype.kind not in 'uif':
        raise ValueError(f'Expected non-empty numeric (H,W) measurement: {path}')
    data = raw.astype(np.float32) / scale
    if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1.0 + 1e-6:
        raise ValueError(f'Measurement must be finite in [0,1]: {path}; check white_level')
    return data


def load_stokes(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    data = np.load(path, allow_pickle=False).astype(np.float32)
    if data.ndim != 3 or data.shape[-1] != 4 or not data.size:
        raise ValueError(f'Expected non-empty (H,W,4) Stokes: {path} ({data.shape})')
    if shape is not None and data.shape[:2] != shape:
        raise ValueError(f'Measurement/label shape mismatch: {path} ({data.shape} vs {shape})')
    if not np.isfinite(data).all():
        raise ValueError(f'Non-finite Stokes values: {path}')
    return data


def stokes_physics(data: np.ndarray, tolerance: float = 1e-5) -> dict:
    """Diagnose raw common-scale Stokes; never silently project labels."""
    if data.ndim != 3 or data.shape[-1] != 4 or not data.size or not np.isfinite(data).all():
        raise ValueError('Expected finite non-empty (H,W,4) Stokes')
    invalid = (data[..., 0] < -tolerance) | (
        np.linalg.norm(data[..., 1:], axis=-1) > data[..., 0] + tolerance)
    return {'pixels': int(invalid.size), 'invalid_pixels': int(invalid.sum()),
            'invalid_fraction': float(invalid.mean()), 'tolerance': tolerance}
