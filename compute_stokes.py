"""compute_stokes.py — 从 5 张偏振照片计算 Stokes 参数 (S0, S1, S2, S3) 并保存为 .npy

原理（基于旋转四分之一波片法）：
    LP 固定在 45°，QWP 分别转到 0°、22.5°、45°、67.5°、90°
    拍摄得到 5 张灰度强度图 I(0), I(22.5), I(45), I(67.5), I(90)

    由 Mueller 矩阵推导，Stokes 参数为：
        S0 = I(0) + I(90)  — 理想光学约定下的总强度
        S1 = 2[I(22.5) - I(67.5)] - sqrt(2)S3                  — 0°/90° 偏振差
        S2 = 2I(45)-S0             — 45°/135° 偏振差
        S3 = I(0) - I(90)         — 圆偏振分量

用法：
    （单个物品）
    py -3 compute_stokes.py --name ruler --raw-dir stokes_raw --out-dir stokes_npy

    （批量处理全部物品）
    py -3 compute_stokes.py --batch --raw-dir stokes_raw --out-dir stokes_npy

文件命名约定：
    stokes_raw/ 文件夹中，每个物品的 5 张照片命名为：
        {物品名}_0.png
        {物品名}_22.5.png
        {物品名}_45.png
        {物品名}_67.5.png
        {物品名}_90.png

    例如：ruler_0.png, ruler_22.5.png, ruler_45.png, ruler_67.5.png, ruler_90.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from polarized_csi.io import load_measurement, stokes_physics
from polarized_csi.experiment import describe_file, save_record


ANGLE_SUFFIXES = ["_0", "_22.5", "_45", "_67.5", "_90"]


def load_gray(path: Path, white_level=None) -> np.ndarray:
    """Read calibrated relative grayscale intensity in [0,1]."""
    return load_measurement(path, white_level)


def find_image(directory: Path, stem: str) -> Path:
    """在目录中查找指定 stem 的图片文件（支持 png/jpg/tif 等格式）。"""
    matches = [p for p in directory.iterdir() if p.is_file() and p.stem == stem
               and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy')]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous angle capture: {directory / stem}')
    if not matches:
        raise FileNotFoundError(f'Missing angle capture: {directory / stem}')
    return matches[0]


def compute_stokes(raw_dir: Path, name: str, white_level=None) -> np.ndarray:
    """从 5 张照片计算 Stokes 参数，返回 (H, W, 4) 数组。"""

    captures = [load_gray(find_image(raw_dir, name + suffix), white_level)
                for suffix in ANGLE_SUFFIXES]
    if len({image.shape for image in captures}) != 1:
        raise ValueError(f'Five-angle images have different shapes: {name}')
    I_0, I_22, I_45, I_67, I_90 = captures
    # Ideal QWP with analyzer at 45 degrees; sign convention requires calibration.
    S0 = I_0 + I_90
    S3 = I_0 - I_90
    S1 = 2.0 * (I_22 - I_67) - np.sqrt(2.0) * S3  # 注意这里是 (I_22 - I_67)
    S2 = 2.0 * I_45 - S0
    # 堆叠为 (H, W, 4)
    stokes = np.stack([S0, S1, S2, S3], axis=-1)
    return stokes


def get_all_names(raw_dir: Path) -> list[str]:
    """自动扫描 raw_dir，找出所有物品名称。

    通过查找以 '_0' 结尾的文件来确定物品名。
    例如 ruler_0.png → 物品名为 'ruler'
    """
    return sorted({p.stem[:-2] for p in raw_dir.iterdir()
                   if p.is_file() and p.stem.endswith('_0') and p.suffix.lower()
                   in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy')})


def main() -> None:
    ap = argparse.ArgumentParser(description="从 5 张偏振照片计算 Stokes 参数")
    ap.add_argument("--name", type=str, default=None,
                    help="单个物品名称（如 ruler）")
    ap.add_argument("--batch", action="store_true",
                    help="批量处理 raw-dir 中所有物品")
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="存放 5 张原始偏振照片的文件夹")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="输出 .npy 文件的文件夹")
    ap.add_argument("--white-level", type=float)
    args = ap.parse_args()
    if bool(args.name) == bool(args.batch):
        ap.error("Choose exactly one of --name or --batch")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        names = get_all_names(args.raw_dir)
        if not names:
            print(f"错误：在 {args.raw_dir} 中没有找到任何物品照片")
            print(f"请确保文件命名格式为：物品名_0.png, 物品名_22.5.png 等")
            ap.error("No angle captures found")
        print(f"找到 {len(names)} 个物品：{names}")
    elif args.name:
        names = [args.name]
    else:
        print("错误：请指定 --name 或 --batch")
        return

    failures = []
    for name in names:
        print(f"正在处理：{name} ...", end=" ")
        try:
            stokes = compute_stokes(args.raw_dir, name, args.white_level)
            out_path = args.out_dir / f"{name}.npy"
            np.save(out_path, stokes)
            sources = [describe_file(find_image(args.raw_dir, name + suffix)) for suffix in ANGLE_SUFFIXES]
            save_record(out_path, {'sources': sources, 'white_level': args.white_level,
                                  'convention': 'ideal QWP, LP 45 degrees; S3 sign needs calibration',
                                  'physics': stokes_physics(stokes)})
            print(f"OK 保存到 {out_path}  形状={stokes.shape}")
        except (FileNotFoundError, ValueError, OSError) as e:
            failures.append(name)
            print(f"FAIL {e}")

    if failures:
        raise SystemExit(f"Failed objects: {failures}")
    print("\n全部完成！")


if __name__ == "__main__":
    main()
