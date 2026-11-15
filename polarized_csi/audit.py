"""Check pair integrity and diagnose physical consistency without training."""
import argparse
import json
from pathlib import Path

from polarized_csi.io import load_measurement, load_stokes, stokes_physics
from polarized_csi.paths import list_pairs


def audit_pairs(pairs, white_level=None):
    samples = []
    for measurement, label in pairs:
        gray = load_measurement(measurement, white_level)
        stokes = load_stokes(label, gray.shape)
        samples.append({'name': measurement.stem, 'shape': list(stokes.shape),
                        'physics': stokes_physics(stokes)})
    pixels = sum(s['physics']['pixels'] for s in samples)
    invalid = sum(s['physics']['invalid_pixels'] for s in samples)
    return {'sample_count': len(samples), 'pixels': pixels, 'invalid_pixels': invalid,
            'invalid_fraction': invalid / pixels, 'samples': samples,
            'interpretation': 'Only valid for raw Stokes channels with one common intensity scale; not proof of acquisition provenance.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compressed', type=Path, required=True)
    ap.add_argument('--stokes', type=Path, required=True)
    ap.add_argument('--white-level', type=float)
    ap.add_argument('--out-json', type=Path)
    ap.add_argument('--strict-physics', action='store_true', help='Fail if any pixel violates ideal Stokes constraints')
    args = ap.parse_args()
    result = audit_pairs(list_pairs(args.compressed, args.stokes), args.white_level)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if args.strict_physics and result['invalid_pixels']:
        raise SystemExit('Physical consistency check failed; investigate calibration/label scale')


if __name__ == '__main__':
    main()
