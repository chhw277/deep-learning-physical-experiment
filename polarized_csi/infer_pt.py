"""Infer four Stokes channels using the linear DL-M1 baseline."""
import argparse
from pathlib import Path

import numpy as np
import torch

from polarized_csi.io import load_measurement
from polarized_csi.experiment import describe_file, save_record
from polarized_csi.model_pt import DL_M1

load_compressed = load_measurement


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--white-level', type=float)
    ap.add_argument('--out-npy', type=Path, default=Path('results/prediction.npy'))
    args = ap.parse_args()
    if args.out_npy.suffix != '.npy':
        ap.error('out-npy must end in .npy so the output and provenance paths agree')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gray = load_measurement(args.input, args.white_level)
    model = DL_M1().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        result = model(torch.from_numpy(gray)[None, None].to(device))[0].permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Prediction contains NaN/Inf; output not written')
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, result.astype(np.float32))
    save_record(args.out_npy, {'framework': 'pytorch', 'shape': list(result.shape),
                             'white_level': args.white_level,
                             'measurement': describe_file(args.input), 'weights': describe_file(args.weights)})
    print(f'Saved {args.out_npy} shape={result.shape}')


if __name__ == '__main__':
    main()
