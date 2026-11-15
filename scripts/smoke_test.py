"""Run an end-to-end software check on generated ideal data, not laboratory results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compute_stokes import compute_stokes, ANGLE_SUFFIXES


def generate_data(root, seed=7):
    """Restricted fixed-polarization family; no general inverse-recovery claim."""
    rng = np.random.default_rng(seed)
    for split, count in [('train', 4), ('val', 2)]:
        raw, cdir, sdir = (root / split / name for name in ('raw', 'c', 's'))
        for folder in (raw, cdir, sdir):
            folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f'{split}{index:02}'
            s0 = rng.uniform(0.2, 0.8, (16, 16)).astype(np.float32)
            stokes = np.stack([s0, .2*s0, -.3*s0, .4*s0], axis=-1)
            for suffix, degrees in zip(ANGLE_SUFFIXES, [0, 22.5, 45, 67.5, 90]):
                angle = np.deg2rad(degrees)
                intensity = .5*(stokes[..., 0] + .5*np.sin(4*angle)*stokes[..., 1]
                                + np.sin(2*angle)**2*stokes[..., 2]
                                + np.cos(2*angle)*stokes[..., 3])
                np.save(raw / (name + suffix + '.npy'), intensity)
            recovered = compute_stokes(raw, name)
            np.testing.assert_allclose(recovered, stokes, atol=2e-7)
            np.save(sdir / (name + '.npy'), recovered)
            # Analyzer at 0 degrees and QWP at 0 degrees: I=(S0+S1)/2.
            np.save(cdir / (name + '.npy'), .5*(stokes[..., 0]+stokes[..., 1]))
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=ROOT/'runs/smoke')
    args = ap.parse_args()
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        ap.error('out-dir must be empty; choose a new directory to preserve earlier runs')
    data = generate_data(output / 'generated')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    commands = [
        ['-m', 'polarized_csi.audit', '--compressed', str(data/'train/c'), '--stokes', str(data/'train/s'), '--strict-physics'],
        ['-m', 'polarized_csi.train_pt', '--train-compressed', str(data/'train/c'), '--train-stokes', str(data/'train/s'),
         '--val-compressed', str(data/'val/c'), '--val-stokes', str(data/'val/s'), '--epochs', '2', '--batch-size', '2',
         '--out', str(output/'model.pt')],
        ['-m', 'polarized_csi.infer_pt', '--weights', str(output/'model.pt'), '--input', str(data/'val/c/val00.npy'),
         '--out-npy', str(output/'prediction.npy')],
        ['evaluate.py', '--pred', str(output/'prediction.npy'), '--true', str(data/'val/s/val00.npy'),
         '--data-range', '1.0', '--out-json', str(output/'evaluation.json')],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    print('PASS: ideal formula, data audit, training, checkpoint, inference and evaluation.')
    print('Generated data only; two epochs and a restricted family do not establish experimental accuracy.')


if __name__ == '__main__':
    main()
