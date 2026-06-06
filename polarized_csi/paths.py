"""数据路径配对（无深度学习框架依赖）。"""

from __future__ import annotations

from pathlib import Path


def list_pairs(
    compressed_dir: str | Path,
    stokes_dir: str | Path,
    exts: tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"),
) -> list[tuple[Path, Path]]:
    """
    同一文件名（不含扩展名）配对：
      compressed_dir / foo.png  <->  stokes_dir / foo.npy
    """
    cdir = Path(compressed_dir)
    sdir = Path(stokes_dir)
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(cdir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        npy = sdir / f"{p.stem}.npy"
        if not npy.is_file():
            continue
        pairs.append((p, npy))
    if not pairs:
        raise FileNotFoundError(
            f"No pairs found under {cdir} / {sdir} (need matching stem.npy)"
        )
    return pairs
