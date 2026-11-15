"""Strict same-stem measurement/label pairing, independent of ML frameworks."""
from pathlib import Path

IMAGE_EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.npy')


def list_pairs(compressed_dir: str | Path, stokes_dir: str | Path,
               exts: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[tuple[Path, Path]]:
    cdir, sdir = Path(compressed_dir), Path(stokes_dir)
    measurements = {}
    for path in sorted(cdir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.stem in measurements:
            raise ValueError(f'Duplicate measurement stem: {path.stem}')
        measurements[path.stem] = path
    labels = {p.stem: p for p in sdir.iterdir() if p.is_file() and p.suffix == '.npy'}
    missing = sorted(measurements.keys() - labels.keys())
    orphaned = sorted(labels.keys() - measurements.keys())
    if missing or orphaned:
        raise ValueError(f'Unpaired data; missing labels={missing}, missing measurements={orphaned}')
    if not measurements:
        raise FileNotFoundError(f'No pairs found under {cdir} / {sdir}')
    return [(measurements[name], labels[name]) for name in sorted(measurements)]


def check_disjoint(train_pairs, validation_pairs) -> None:
    """Reject shared file paths; object-level grouping needs a capture manifest."""
    train_paths = {p.resolve() for pair in train_pairs for p in pair}
    overlap = train_paths & {p.resolve() for pair in validation_pairs for p in pair}
    if overlap:
        raise ValueError(f'Train/validation files overlap: {sorted(map(str, overlap))}')
