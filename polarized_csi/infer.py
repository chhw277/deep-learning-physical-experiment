"""对单张压缩图推理，输出 (H,W,4) 斯托克斯 numpy。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from polarized_csi.io import load_measurement
from polarized_csi.experiment import describe_file, save_record
from tensorflow import keras


def load_compressed(path: Path, white_level=None) -> np.ndarray:
    return load_measurement(path, white_level)[None, ..., None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, default=Path("stokes_pred.npy"))
    ap.add_argument("--white-level", type=float)
    args = ap.parse_args()
    if args.out_npy.suffix != '.npy':
        ap.error('out-npy must end in .npy so the output and provenance paths agree')

    model = keras.models.load_model(str(args.weights), compile=False)
    x = load_compressed(args.input, args.white_level)
    y = model.predict(x, verbose=0)[0]
    if not np.isfinite(y).all():
        raise FloatingPointError('Prediction contains NaN/Inf; output not written')
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out_npy), y.astype(np.float32))
    save_record(args.out_npy, {'framework': 'tensorflow', 'shape': list(y.shape),
                              'white_level': args.white_level,
                              'measurement': describe_file(args.input), 'weights': describe_file(args.weights)})
    print(f"Saved {args.out_npy} shape={y.shape}")


if __name__ == "__main__":
    main()
